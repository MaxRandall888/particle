#include "octree.hpp"
#include "particle.hpp"
#include "physics.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <getopt.h>

struct SceneObject {
    std::string name, model, velocityMode;
    std::size_t particleCount{};
    double mass{}, scale{}, height{}, bulgeRadius{}, bulgeFraction{}, toomreQ{};
    std::uint32_t seed{};
    std::array<double, 3> position{}, velocity{}, rotation{}; // Rotation in degrees, X then Y then Z.
};

struct SimulationParameters {
    std::size_t particleCount = 1000;
    int stepCount = 2000;
    double timestep = 0.001;
    std::string integrator = "euler";
    double theta = 0.5;
    double softeningLength = 0.02;
    int threadCount = 16;
    int snapshotInterval = 10;
    int frameCount = 0; // Zero selects the legacy interval-based schedule.
    int maxDepth = 30;
    std::uint32_t seed = 42;
    double lengthScaleKpc = 10.0;
    double diskScaleLengthKpc = 3.0;
    double diskScaleHeightKpc = 0.6; // sech^2(z / z0); exponential tails have height z0/2.
    double bulgeScaleRadiusKpc = 1.0;
    double bulgeMassFraction = 0.2;
    double diskToomreQ = 1.5; // Set at R = 2.2 disk scale lengths, not at every radius.
    double massScaleSolarMasses = 1e11;
    std::filesystem::path outputDirectory = "output/run_001";
    std::filesystem::path sceneFile, initialStateFile;
    std::vector<SceneObject> objects;
    bool prepareOnly = false;
};

using Clock = std::chrono::steady_clock;

template <typename Number>
void parseNumber(std::string_view text, Number& value, std::string_view name) {
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::invalid_argument("Invalid number for " + std::string(name));
    }
}

double secondsSince(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::size_t bulgeParticleCount(const SimulationParameters& parameters) {
    return static_cast<std::size_t>(std::llround(parameters.particleCount * parameters.bulgeMassFraction));
}

// Smooth stellar model used only to initialize velocities (G = total mass = 1).
// Evolution still uses the existing softened N-body forces, not this potential.
struct GalaxyModel {
    double diskMass, bulgeMass;
    double diskLength, diskHeight, bulgeRadius, softening;

    double surfaceDensity(double radius) const {
        return diskMass * std::exp(-radius / diskLength) /
               (2.0 * std::numbers::pi * diskLength * diskLength);
    }

    double circularSpeedSquared(double radius) const {
        // Freeman exponential thin-disk curve, with a softened radial argument.
        // This is an approximation to our finite-height, softened particle disk.
        const double softenedRadius = std::hypot(radius, softening);
        const double y = softenedRadius / (2.0 * diskLength);
        const double bessel = std::cyl_bessel_i(0, y) * std::cyl_bessel_k(0, y) -
                              std::cyl_bessel_i(1, y) * std::cyl_bessel_k(1, y);
        const double disk = diskMass * radius * radius * std::max(0.0, bessel) /
                            (2.0 * diskLength * diskLength * diskLength);
        const double bulgeDistance = std::sqrt(radius * radius + bulgeRadius * bulgeRadius +
                                              softening * softening);
        return disk + bulgeMass * radius * radius /
                      (bulgeDistance * bulgeDistance * bulgeDistance);
    }

    double epicyclicRatio(double radius) const {
        // kappa^2 / (4 Omega^2) = (2 + d ln(vc^2) / d ln(R)) / 4.
        const double step = 1e-3;
        const double slope = (std::log(circularSpeedSquared(radius * std::exp(step))) -
                              std::log(circularSpeedSquared(radius * std::exp(-step)))) /
                             (2.0 * step);
        return std::max(1e-6, (2.0 + slope) / 4.0);
    }

    double sphericalAcceleration(double radius) const {
        // Angular-average approximation for the bulge Jeans calculation.
        // The thin disk's mass inside a sphere is M[1 - (1+r/Rd)exp(-r/Rd)].
        const double distance = std::hypot(radius, softening);
        const double x = distance / diskLength;
        const double diskEnclosed = diskMass * (-std::expm1(-x) - x * std::exp(-x));
        const double bulgeDistance = std::sqrt(radius * radius + bulgeRadius * bulgeRadius +
                                              softening * softening);
        return diskEnclosed * radius / (distance * distance * distance) +
               bulgeMass * radius / (bulgeDistance * bulgeDistance * bulgeDistance);
    }

    double relativePotential(double radius) const {
        // Matching spherical escape-speed estimate; zero potential at infinity.
        const double distance = std::hypot(radius, softening);
        return diskMass * (-std::expm1(-distance / diskLength)) / distance +
               bulgeMass / std::sqrt(radius * radius + bulgeRadius * bulgeRadius +
                                     softening * softening);
    }

    double bulgeDensityShape(double radius) const {
        return std::pow(1.0 + radius * radius / (bulgeRadius * bulgeRadius), -2.5);
    }
};

void initializeParticles(std::vector<Particle>& particles, const SimulationParameters& parameters) {
    particles.assign(parameters.particleCount, Particle{});
    if (particles.empty()) {
        return;
    }

    const std::size_t bulgeCount = bulgeParticleCount(parameters);
    const std::size_t diskCount = particles.size() - bulgeCount;
    const double mass = 1.0 / static_cast<double>(particles.size());
    const GalaxyModel model{diskCount * mass, bulgeCount * mass,
                            parameters.diskScaleLengthKpc / parameters.lengthScaleKpc,
                            parameters.diskScaleHeightKpc / parameters.lengthScaleKpc,
                            parameters.bulgeScaleRadiusKpc / parameters.lengthScaleKpc,
                            parameters.softeningLength};
    std::mt19937 generator(parameters.seed);
    // Open interval: logarithms, atanh and inverse CDFs never see 0 or 1.
    auto uniform = [&]() { return (static_cast<double>(generator()) + 0.5) / 4294967296.0; };
    std::normal_distribution<double> normal(0.0, 1.0);

    // sigma_R^2 falls exponentially. Normalize Q at 2.2 Rd, where a stellar
    // disk's rotation curve is near its peak; Q is not globally constant.
    const double referenceRadius = 2.2 * model.diskLength;
    const double kappa = 2.0 * std::sqrt(model.circularSpeedSquared(referenceRadius) *
                                      model.epicyclicRatio(referenceRadius)) / referenceRadius;
    const double referenceDispersion = parameters.diskToomreQ * 3.36 *
                                       model.surfaceDensity(referenceRadius) / kappa;
    const double radialVariance0 = referenceDispersion * referenceDispersion * std::exp(2.2);

    // Height-averaged vertical Jeans moment <z dPhi/dz>. Equal-probability
    // quadrature samples the sech^2 profile. Include softening in the local
    // infinite-sheet force instead of assuming an unsoftened, harmonic core.
    constexpr int heightSamples = 64;
    std::array<double, heightSamples> heights{};
    for (int i = 0; i < heightSamples; ++i) {
        heights[i] = model.diskHeight * std::atanh(2.0 * (i + 0.5) / heightSamples - 1.0);
    }
    double sheetMoment = 0.0;
    for (double z : heights) {
        for (double sourceZ : heights) {
            const double dz = z - sourceZ;
            sheetMoment += z * dz / std::hypot(dz, model.softening) /
                           (heightSamples * heightSamples);
        }
    }

    // Isotropic bulge Jeans moments: sigma^2 = integral_r^infinity(rho*g ds)/rho.
    // Tabulate once on a log grid, rather than integrating for every particle.
    constexpr int tableSize = 1024;
    std::array<double, tableSize> bulgeVariance{};
    const double radiusMin = model.bulgeRadius * 1e-4;
    const double radiusMax = model.bulgeRadius * 1e4;
    const double logStep = std::log(radiusMax / radiusMin) / (tableSize - 1);
    double upperRadius = radiusMax;
    double upperIntegrand = model.bulgeDensityShape(upperRadius) * model.sphericalAcceleration(upperRadius);
    double integral = model.bulgeDensityShape(radiusMax) / (6.0 * radiusMax);
    bulgeVariance.back() = integral / model.bulgeDensityShape(radiusMax);
    for (int i = tableSize - 2; i >= 0; --i) {
        const double radius = radiusMin * std::exp(i * logStep);
        const double integrand = model.bulgeDensityShape(radius) * model.sphericalAcceleration(radius);
        integral += 0.5 * (upperIntegrand + integrand) * (upperRadius - radius);
        bulgeVariance[i] = integral / model.bulgeDensityShape(radius);
        upperRadius = radius;
        upperIntegrand = integrand;
    }

    for (std::size_t i = 0; i < particles.size(); ++i) {
        Particle& particle = particles[i];
        particle.mass = mass;
        const double theta = 2.0 * std::numbers::pi * uniform();
        if (i < diskCount) {
            // p(R) = R exp(-R/Rd)/Rd^2 includes the cylindrical area Jacobian.
            const double radius = -model.diskLength * std::log(uniform() * uniform());
            particle.x = radius * std::cos(theta);
            particle.y = radius * std::sin(theta);
            particle.z = model.diskHeight * std::atanh(2.0 * uniform() - 1.0);

            const double radialVariance = radialVariance0 * std::exp(-radius / model.diskLength);
            // Radial Jeans equation with no tilt term, including asymmetric drift.
            const double azimuthalSecondMoment = model.circularSpeedSquared(radius) +
                radialVariance * (1.0 - 2.0 * radius / model.diskLength);
            if (azimuthalSecondMoment < 0.0) {
                throw std::runtime_error("Galaxy dispersions are incompatible with radial Jeans balance");
            }
            // The epicyclic approximation fails in the pressure-dominated core.
            // There, assign the allowed azimuthal moment to dispersion, not rotation.
            const double azimuthalVariance = std::min(radialVariance * model.epicyclicRatio(radius),
                                                       azimuthalSecondMoment);
            const double streamingSpeed = std::sqrt(azimuthalSecondMoment - azimuthalVariance);
            double verticalVariance = 2.0 * std::numbers::pi * model.surfaceDensity(radius) * sheetMoment;
            for (double z : heights) {
                const double bulgeDistance = std::sqrt(radius * radius + z * z +
                    model.bulgeRadius * model.bulgeRadius + model.softening * model.softening);
                verticalVariance += model.bulgeMass * z * z /
                    (heightSamples * bulgeDistance * bulgeDistance * bulgeDistance);
            }
            const double escapeSquared = 2.0 * model.relativePotential(std::hypot(radius, particle.z));
            double radialVelocity, azimuthalVelocity;
            do {
                radialVelocity = std::sqrt(radialVariance) * normal(generator);
                azimuthalVelocity = streamingSpeed + std::sqrt(azimuthalVariance) * normal(generator);
                particle.vz = std::sqrt(verticalVariance) * normal(generator);
            } while (radialVelocity * radialVelocity + azimuthalVelocity * azimuthalVelocity +
                     particle.vz * particle.vz >= escapeSquared);
            particle.vx = radialVelocity * std::cos(theta) - azimuthalVelocity * std::sin(theta);
            particle.vy = radialVelocity * std::sin(theta) + azimuthalVelocity * std::cos(theta);
        } else {
            // Plummer cumulative mass: M(<r)/Mb = r^3/(r^2+b^2)^(3/2).
            const double fraction = std::pow(uniform(), 2.0 / 3.0);
            const double radius = model.bulgeRadius * std::sqrt(fraction / (1.0 - fraction));
            const double cosPolar = 2.0 * uniform() - 1.0;
            const double sinPolar = std::sqrt(1.0 - cosPolar * cosPolar);
            particle.x = radius * sinPolar * std::cos(theta);
            particle.y = radius * sinPolar * std::sin(theta);
            particle.z = radius * cosPolar;
            double variance;
            if (radius >= radiusMax) {
                variance = 1.0 / (6.0 * radius);
            } else {
                const double offset = std::max(0.0, std::log(radius / radiusMin) / logStep);
                const int index = std::min(static_cast<int>(offset), tableSize - 2);
                variance = std::lerp(bulgeVariance[index], bulgeVariance[index + 1], offset - index);
            }
            const double escapeSquared = 2.0 * model.relativePotential(radius);
            do {
                particle.vx = std::sqrt(variance) * normal(generator);
                particle.vy = std::sqrt(variance) * normal(generator);
                particle.vz = std::sqrt(variance) * normal(generator);
            } while (particle.vx * particle.vx + particle.vy * particle.vy +
                     particle.vz * particle.vz >= escapeSquared);
        }
    }

    // Remove finite-sample center-of-mass offsets and bulk drift; all particles
    // keep their relative positions/velocities and their original array indices.
    Particle mean{};
    for (const Particle& particle : particles) {
        mean.x += particle.x * mass; mean.y += particle.y * mass; mean.z += particle.z * mass;
        mean.vx += particle.vx * mass; mean.vy += particle.vy * mass; mean.vz += particle.vz * mass;
    }
    for (Particle& particle : particles) {
        particle.x -= mean.x; particle.y -= mean.y; particle.z -= mean.z;
        particle.vx -= mean.vx; particle.vy -= mean.vy; particle.vz -= mean.vz;
    }
}

double normalizeInitialVirial(std::vector<Particle>& particles, std::vector<Node>& nodes,
                             std::vector<std::int32_t>& nextParticle,
                             const SimulationParameters& parameters) {
    if (particles.size() < 2) {
        return 1.0;
    }
    buildOctree(particles, nodes, nextParticle, parameters.maxDepth);
    aggregateMass(0, particles, nodes, nextParticle);
    calculateAccelerations(particles, nodes, nextParticle, parameters.theta,
                           parameters.softeningLength, parameters.threadCount);
    double twiceKinetic = 0.0;
    double forceVirial = 0.0;
    for (const Particle& particle : particles) {
        twiceKinetic += particle.mass * (particle.vx * particle.vx + particle.vy * particle.vy +
                                         particle.vz * particle.vz);
        forceVirial += particle.mass * (particle.x * particle.ax + particle.y * particle.ay +
                                        particle.z * particle.az);
    }
    // For softened gravity the force virial differs from potential energy.
    // This enforces global 2T + sum(m r.a) = 0, not exact local equilibrium.
    if (twiceKinetic <= 0.0 || forceVirial >= 0.0) {
        return 1.0;
    }
    const double scale = std::sqrt(-forceVirial / twiceKinetic);
    for (Particle& particle : particles) {
        particle.vx *= scale;
        particle.vy *= scale;
        particle.vz *= scale;
    }
    return scale;
}

std::vector<SceneObject> readScene(const std::filesystem::path& filename) {
    std::ifstream file(filename);
    std::string magic;
    int version = 0;
    if (!(file >> magic >> version) || magic != "NBODY_SCENE" || version != 1) {
        throw std::runtime_error("Expected an NBODY_SCENE 1 configuration");
    }
    std::vector<SceneObject> objects;
    while (file >> std::ws && file.peek() != std::char_traits<char>::eof()) {
        SceneObject object;
        file >> object.model >> object.velocityMode >> object.particleCount >> object.mass
             >> object.scale >> object.height >> object.bulgeRadius >> object.bulgeFraction
             >> object.toomreQ >> object.seed;
        for (double& value : object.position) file >> value;
        for (double& value : object.velocity) file >> value;
        for (double& value : object.rotation) file >> value;
        file >> std::quoted(object.name);
        if (!file || (object.model != "uniform3d" && object.model != "gaussian2d" &&
                      object.model != "gaussian3d" && object.model != "galaxy") ||
            (object.velocityMode != "rest" && object.velocityMode != "random" &&
             object.velocityMode != "rotation" && object.velocityMode != "equilibrium") ||
            (object.model == "galaxy") != (object.velocityMode == "equilibrium") ||
            object.particleCount == 0 || object.particleCount > std::numeric_limits<std::int32_t>::max() ||
            !std::isfinite(object.mass) || object.mass <= 0 ||
            !std::isfinite(object.scale) || object.scale <= 0 ||
            !std::isfinite(object.height) || object.height <= 0 ||
            !std::isfinite(object.bulgeRadius) || object.bulgeRadius <= 0 ||
            !std::isfinite(object.bulgeFraction) || object.bulgeFraction < 0 || object.bulgeFraction > 1 ||
            !std::isfinite(object.toomreQ) || object.toomreQ <= 0) {
            throw std::runtime_error("Invalid object in scene configuration");
        }
        for (const auto& values : {object.position, object.velocity, object.rotation}) {
            for (double value : values) {
                if (!std::isfinite(value)) throw std::runtime_error("Scene transforms must be finite");
            }
        }
        objects.push_back(object);
    }
    if (objects.empty()) throw std::runtime_error("Scene must contain at least one object");
    return objects;
}

void rotateVector(double& x, double& y, double& z, const std::array<double, 3>& degrees) {
    for (int axis = 0; axis < 3; ++axis) {
        const double angle = degrees[axis] * (std::numbers::pi / 180.0);
        const double c = std::cos(angle), s = std::sin(angle);
        if (axis == 0) { const double old = y; y = c * y - s * z; z = s * old + c * z; }
        if (axis == 1) { const double old = x; x = c * x + s * z; z = -s * old + c * z; }
        if (axis == 2) { const double old = x; x = c * x - s * y; y = s * old + c * y; }
    }
}

void initializeScene(std::vector<Particle>& particles, const SimulationParameters& parameters) {
    particles.clear();
    particles.reserve(parameters.particleCount);
    for (const SceneObject& object : parameters.objects) {
        SimulationParameters local = parameters;
        local.objects.clear();
        local.particleCount = object.particleCount;
        local.seed = object.seed;
        local.diskScaleLengthKpc = object.scale * local.lengthScaleKpc;
        local.diskScaleHeightKpc = object.height * local.lengthScaleKpc;
        local.bulgeScaleRadiusKpc = object.bulgeRadius * local.lengthScaleKpc;
        local.bulgeMassFraction = object.bulgeFraction;
        local.diskToomreQ = object.toomreQ;
        std::vector<Particle> component;
        if (object.model == "galaxy") {
            initializeParticles(component, local);
        } else {
            component.assign(object.particleCount, Particle{});
            std::mt19937 generator(object.seed);
            std::uniform_real_distribution<double> uniform(-object.scale, object.scale);
            std::normal_distribution<double> normal(0.0, 1.0);
            Particle mean{};
            for (Particle& particle : component) {
                particle.x = object.model == "uniform3d" ? uniform(generator) : object.scale * normal(generator);
                particle.y = object.model == "uniform3d" ? uniform(generator) : object.scale * normal(generator);
                particle.z = object.model == "gaussian2d" ? 0.0 :
                    (object.model == "uniform3d" ? uniform(generator) : object.scale * normal(generator));
                particle.mass = 1.0 / object.particleCount;
                if (object.velocityMode == "random") {
                    particle.vx = normal(generator);
                    particle.vy = normal(generator);
                    particle.vz = object.model == "gaussian2d" ? 0.0 : normal(generator);
                } else if (object.velocityMode == "rotation") {
                    const double radius = std::hypot(particle.x, particle.y);
                    const double r = std::hypot(radius, particle.z);
                    const double q = r / object.scale;
                    const double enclosed = object.model == "gaussian2d" ? -std::expm1(-q * q / 2.0) :
                        (object.model == "gaussian3d" ? std::max(0.0, std::erf(q / std::sqrt(2.0)) -
                            std::sqrt(2.0 / std::numbers::pi) * q * std::exp(-q * q / 2.0)) :
                         std::min(1.0, std::pow(q / std::sqrt(3.0), 3)));
                    const double distance = std::hypot(r, parameters.softeningLength);
                    const double angularSpeed = std::sqrt(enclosed / (distance * distance * distance));
                    particle.vx = -angularSpeed * particle.y;
                    particle.vy = angularSpeed * particle.x;
                }
                mean.x += particle.x * particle.mass; mean.y += particle.y * particle.mass;
                mean.z += particle.z * particle.mass; mean.vx += particle.vx * particle.mass;
                mean.vy += particle.vy * particle.mass; mean.vz += particle.vz * particle.mass;
            }
            for (Particle& particle : component) {
                particle.x -= mean.x; particle.y -= mean.y; particle.z -= mean.z;
                particle.vx -= mean.vx; particle.vy -= mean.vy; particle.vz -= mean.vz;
            }
        }
        for (Particle& particle : component) {
            particle.mass *= object.mass;
        }
        if (object.velocityMode != "rest") {
            std::vector<Node> nodes;
            std::vector<std::int32_t> links;
            normalizeInitialVirial(component, nodes, links, local);
        }
        // Normalize each object before adding orbital/bulk velocities. Do not
        // virialize a multi-galaxy encounter as one combined bound system.
        for (Particle& particle : component) {
            rotateVector(particle.x, particle.y, particle.z, object.rotation);
            rotateVector(particle.vx, particle.vy, particle.vz, object.rotation);
            particle.x += object.position[0]; particle.y += object.position[1]; particle.z += object.position[2];
            particle.vx += object.velocity[0]; particle.vy += object.velocity[1]; particle.vz += object.velocity[2];
            particles.push_back(particle);
        }
    }
}

void writeInitialState(const std::vector<Particle>& particles, const std::filesystem::path& filename) {
    // Private preparation format, separate from unchanged float32 .npy snapshots.
    static_assert(std::endian::native == std::endian::little && sizeof(double) == 8);
    std::ofstream file(filename, std::ios::binary);
    file.write("NBODYIC1", 8);
    const std::uint64_t count = particles.size();
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const Particle& p : particles) {
        const double row[]{p.x, p.y, p.z, p.vx, p.vy, p.vz, p.mass};
        for (double value : row) {
            if (!std::isfinite(value)) throw std::runtime_error("Initial state contains nonfinite values");
        }
        file.write(reinterpret_cast<const char*>(row), sizeof(row));
    }
    file.close();
    if (!file) throw std::runtime_error("Cannot write prepared initial state");
}

void readInitialState(std::vector<Particle>& particles, const std::filesystem::path& filename) {
    std::ifstream file(filename, std::ios::binary);
    std::array<char, 8> magic{};
    std::uint64_t count = 0;
    file.read(magic.data(), magic.size());
    file.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!file || std::string_view(magic.data(), magic.size()) != "NBODYIC1" || count == 0 ||
        count > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
        std::filesystem::file_size(filename) != 16 + count * 7 * sizeof(double)) {
        throw std::runtime_error("Invalid prepared initial state");
    }
    particles.assign(count, Particle{});
    for (Particle& p : particles) {
        double row[7];
        file.read(reinterpret_cast<char*>(row), sizeof(row));
        for (double value : row) {
            if (!std::isfinite(value)) throw std::runtime_error("Prepared state contains nonfinite values");
        }
        if (!file || row[6] <= 0) throw std::runtime_error("Invalid prepared particle mass");
        p.x = row[0]; p.y = row[1]; p.z = row[2];
        p.vx = row[3]; p.vy = row[4]; p.vz = row[5]; p.mass = row[6];
    }
}

std::string jsonString(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
        else out << c;
    }
    out << '"';
    return out.str();
}

std::string frameFilename(std::size_t frameIndex) {
    std::ostringstream name;
    name << "frame_" << std::setfill('0') << std::setw(6) << frameIndex << ".npy";
    return name.str();
}

void writeSnapshot(const std::vector<Particle>& particles,
                   const std::filesystem::path& filename, std::vector<float>& positions) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    static_assert(std::endian::native == std::endian::little,
                  "The snapshot writer currently targets little-endian systems");

    positions.resize(3 * particles.size());
    for (std::size_t p = 0; p < particles.size(); ++p) {
        const double coordinates[3]{particles[p].x, particles[p].y, particles[p].z};
        for (int axis = 0; axis < 3; ++axis) {
            const double value = coordinates[axis];
            if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max()) {
                throw std::runtime_error("Snapshot position is not representable as finite float32");
            }
            positions[3 * p + axis] = static_cast<float>(value);
        }
    }

    // NPY 1.0: eight-byte magic/version, two-byte header length, then an ASCII
    // dictionary padded so that the binary payload starts on a 64-byte boundary.
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                         std::to_string(particles.size()) + ", 3), }";
    const std::size_t padding = (64 - (10 + header.size() + 1) % 64) % 64;
    header.append(padding, ' ');
    header += '\n';
    const auto headerLength = static_cast<std::uint16_t>(header.size());
    const char lengthBytes[2]{static_cast<char>(headerLength & 0xff),
                              static_cast<char>((headerLength >> 8) & 0xff)};

    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open snapshot: " + filename.string());
    }
    file.write("\x93NUMPY\x01\x00", 8);
    file.write(lengthBytes, 2);
    file.write(header.data(), static_cast<std::streamsize>(header.size()));
    file.write(reinterpret_cast<const char*>(positions.data()),
               static_cast<std::streamsize>(positions.size() * sizeof(float)));
    file.close();
    if (!file) {
        throw std::runtime_error("Cannot finish writing snapshot: " + filename.string());
    }
}

void writeMetadata(const SimulationParameters& parameters, const std::vector<int>& snapshotSteps,
                   double initialVelocityScale) {
    const auto filename = parameters.outputDirectory / "metadata.json";
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open metadata: " + filename.string());
    }
    file << std::setprecision(17)
         << "{\n"
         << "  \"particle_count\": " << parameters.particleCount << ",\n"
         << "  \"step_count\": " << parameters.stepCount << ",\n"
         << "  \"timestep\": " << parameters.timestep << ",\n"
         << "  \"final_time\": " << parameters.stepCount * parameters.timestep << ",\n"
         << "  \"opening_angle\": " << parameters.theta << ",\n"
         << "  \"softening_length\": " << parameters.softeningLength << ",\n"
         << "  \"thread_count\": " << parameters.threadCount << ",\n"
         << "  \"snapshot_interval\": " << parameters.snapshotInterval << ",\n"
         << "  \"snapshot_sampling\": \"" << (parameters.frameCount > 0 ? "frame_count" : "interval") << "\",\n"
         << "  \"requested_frames\": " << parameters.frameCount << ",\n"
         << "  \"max_tree_depth\": " << parameters.maxDepth << ",\n"
         << "  \"seed\": " << parameters.seed << ",\n"
         << "  \"integrator\": \"" << (parameters.integrator == "euler" ? "explicit_euler" : "leapfrog") << "\",\n"
         << "  \"softening_model\": \"plummer\",\n";
    if (!parameters.objects.empty() || !parameters.initialStateFile.empty()) {
        file << "  \"initial_conditions\": {\"distribution\": \"scene\", \"prepared_state_loaded\": "
             << (parameters.initialStateFile.empty() ? "false" : "true") << ", \"objects\": [\n";
        for (std::size_t i = 0; i < parameters.objects.size(); ++i) {
            const SceneObject& object = parameters.objects[i];
            file << "    {\"name\": " << jsonString(object.name) << ", \"model\": " << jsonString(object.model)
                 << ", \"velocity_mode\": " << jsonString(object.velocityMode)
                 << ", \"particle_count\": " << object.particleCount << ", \"mass\": " << object.mass
                 << ", \"scale\": " << object.scale << ", \"height\": " << object.height
                 << ", \"bulge_radius\": " << object.bulgeRadius << ", \"bulge_fraction\": " << object.bulgeFraction
                 << ", \"toomre_Q_target\": " << object.toomreQ << ", \"seed\": " << object.seed;
            for (const auto& item : {std::pair{"position", object.position}, std::pair{"velocity", object.velocity},
                                    std::pair{"rotation_degrees_xyz", object.rotation}}) {
                file << ", " << jsonString(item.first) << ": [" << item.second[0] << ", "
                     << item.second[1] << ", " << item.second[2] << ']';
            }
            file << '}' << (i + 1 < parameters.objects.size() ? "," : "") << '\n';
        }
        file << "  ]},\n";
    } else {
        file << "  \"initial_conditions\": {\n"
         << "    \"distribution\": \"exponential_sech2_disk_and_plummer_bulge\", \"total_mass\": 1,\n"
         << "    \"disk_particle_count\": " << parameters.particleCount - bulgeParticleCount(parameters) << ",\n"
         << "    \"bulge_particle_count\": " << bulgeParticleCount(parameters) << ",\n"
         << "    \"component_order\": [\"disk\", \"bulge\"],\n"
         << "    \"disk_scale_length\": " << parameters.diskScaleLengthKpc / parameters.lengthScaleKpc << ",\n"
         << "    \"disk_sech2_height\": " << parameters.diskScaleHeightKpc / parameters.lengthScaleKpc << ",\n"
         << "    \"bulge_plummer_radius\": " << parameters.bulgeScaleRadiusKpc / parameters.lengthScaleKpc << ",\n"
         << "    \"disk_scale_length_kpc\": " << parameters.diskScaleLengthKpc << ",\n"
         << "    \"disk_sech2_height_kpc\": " << parameters.diskScaleHeightKpc << ",\n"
         << "    \"bulge_plummer_radius_kpc\": " << parameters.bulgeScaleRadiusKpc << ",\n"
         << "    \"disk_toomre_Q_target_before_virial_scaling\": " << parameters.diskToomreQ << ",\n"
         << "    \"disk_toomre_Q_reference_radius\": "
         << 2.2 * parameters.diskScaleLengthKpc / parameters.lengthScaleKpc << ",\n"
         << "    \"initial_velocity_scale_for_global_virial_balance\": " << initialVelocityScale << ",\n"
         << "    \"velocity_model\": \"approximate_jeans_moments_escape_truncated_gaussians\",\n"
         << "    \"disk_rotation_curve\": \"softened_thin_exponential_disk_plus_plummer_bulge\",\n"
         << "    \"bulge_velocity_model\": \"isotropic_spherical_jeans_disk_monopole\",\n"
         << "    \"disk_vertical_moments\": \"height_averaged_softened_sheet_plus_bulge\",\n"
         << "    \"rotation_direction\": \"counterclockwise_from_positive_z\",\n"
         << "    \"center_of_mass_removed\": true, \"bulk_velocity_removed\": true,\n"
         << "    \"equilibrium\": \"approximate\", \"dark_matter_halo\": false\n"
         << "  },\n";
    }
    file << "  \"units\": {\n"
         << "    \"G\": 1, \"length\": \"L0\", \"mass\": \"M0\", \"time\": \"T0\",\n"
         << "    \"length_scale_kpc\": " << parameters.lengthScaleKpc << ",\n"
         << "    \"mass_scale_solar_masses\": " << parameters.massScaleSolarMasses << ",\n"
         << "    \"time_scale_definition\": \"T0 = sqrt(L0^3 / (G_physical * M0))\"\n"
         << "  },\n"
         << "  \"snapshot_dtype\": \"float32\",\n"
         << "  \"snapshot_shape\": [" << parameters.particleCount << ", 3],\n"
         << "  \"snapshots\": [\n";
    for (std::size_t frame = 0; frame < snapshotSteps.size(); ++frame) {
        file << "    {\"file\": \"" << frameFilename(frame) << "\", \"step\": "
             << snapshotSteps[frame] << ", \"time\": " << snapshotSteps[frame] * parameters.timestep
             << "}" << (frame + 1 < snapshotSteps.size() ? "," : "") << '\n';
    }
    file << "  ]\n}\n";
    file.close();
    if (!file) {
        throw std::runtime_error("Cannot finish writing metadata: " + filename.string());
    }
}

int main(int argc, char* argv[]) {
    const SimulationParameters defaults;
    SimulationParameters parameters = defaults;
    try {
        enum {
            SnapshotInterval = 1000, Timestep, OpeningAngle, Softening, Seed, Scene, PrepareOnly, InitialState, Frames, Integrator
        };
        const option options[]{
            {"threads", required_argument, nullptr, 't'},
            {"steps", required_argument, nullptr, 's'},
            {"snapshot-interval", required_argument, nullptr, SnapshotInterval},
            {"frames", required_argument, nullptr, Frames},
            {"dt", required_argument, nullptr, Timestep},
            {"integrator", required_argument, nullptr, Integrator},
            {"theta", required_argument, nullptr, OpeningAngle},
            {"softening", required_argument, nullptr, Softening},
            {"seed", required_argument, nullptr, Seed},
            {"scene", required_argument, nullptr, Scene},
            {"prepare-only", no_argument, nullptr, PrepareOnly},
            {"initial-state", required_argument, nullptr, InitialState},
            {"output", required_argument, nullptr, 'o'},
            {"help", no_argument, nullptr, 'h'},
            {nullptr, 0, nullptr, 0}
        };
        opterr = 0;
        int choice;
        while ((choice = getopt_long(argc, argv, "t:s:o:h", options, nullptr)) != -1) {
            switch (choice) {
            case 't':
                parseNumber(optarg, parameters.threadCount, "--threads");
                break;
            case 's':
                parseNumber(optarg, parameters.stepCount, "--steps");
                break;
            case SnapshotInterval:
                parseNumber(optarg, parameters.snapshotInterval, "--snapshot-interval");
                break;
            case Frames:
                parseNumber(optarg, parameters.frameCount, "--frames");
                if (parameters.frameCount <= 0) throw std::invalid_argument("--frames must be positive");
                break;
            case Timestep:
                parseNumber(optarg, parameters.timestep, "--dt");
                break;
            case Integrator:
                parameters.integrator = optarg;
                if (parameters.integrator != "euler" && parameters.integrator != "leapfrog") {
                    throw std::invalid_argument("--integrator must be euler or leapfrog");
                }
                break;
            case OpeningAngle:
                parseNumber(optarg, parameters.theta, "--theta");
                break;
            case Softening:
                parseNumber(optarg, parameters.softeningLength, "--softening");
                break;
            case Seed:
                parseNumber(optarg, parameters.seed, "--seed");
                break;
            case Scene:
                parameters.sceneFile = optarg;
                break;
            case PrepareOnly:
                parameters.prepareOnly = true;
                break;
            case InitialState:
                parameters.initialStateFile = optarg;
                break;
            case 'o':
                parameters.outputDirectory = optarg;
                break;
            case 'h':
                std::cout << "Usage: " << argv[0] << " [PARTICLES] [OPTIONS]\n"
                          << "  PARTICLES              Positive particle count (default: "
                          << defaults.particleCount << ")\n"
                          << "  -t, --threads N        Positive thread count (default: "
                          << defaults.threadCount << ")\n"
                          << "  -s, --steps N          Nonnegative integration steps (default: "
                          << defaults.stepCount << ")\n"
                          << "  --snapshot-interval N  Save every N steps; N > 0 (default: "
                          << defaults.snapshotInterval << ")\n"
                          << "  --frames N             Save exactly N evenly spaced snapshots, including initial/final;\n"
                          << "                         overrides interval; 2 <= N <= steps+1 (1 when steps=0)\n"
                          << "  --dt VALUE             Positive timestep in simulation units (default: "
                          << defaults.timestep << ")\n"
                          << "  --integrator METHOD    euler or leapfrog; fixed timestep (default: euler)\n"
                          << "  --theta VALUE          Nonnegative opening angle; 0 disables approximation (default: "
                          << defaults.theta << ")\n"
                          << "  --softening VALUE      Positive Plummer softening length (default: "
                          << defaults.softeningLength << ")\n"
                          << "  --seed N               Unsigned 32-bit random seed (default: "
                          << defaults.seed << ")\n"
                          << "  --scene FILE           Scene configuration generated by the Qt application\n"
                          << "  --prepare-only         Save frame zero, metadata and exact initial_state.bin; no integration\n"
                          << "  --initial-state FILE   Start from a prepared float64 position/velocity/mass state\n"
                          << "  -o, --output DIRECTORY New run directory (default: "
                          << defaults.outputDirectory.string() << ")\n"
                          << "  -h, --help             Show this help and exit\n"
                          << "Initial and final snapshots are always saved. Existing output directories are preserved.\n";
                return 0;
            default:
                throw std::invalid_argument("Unknown option or missing option value; use --help");
            }
        }
        if (argc - optind > 1) {
            throw std::invalid_argument("Expected at most one positional particle count; use --help");
        }
        if (optind < argc) {
            parseNumber(argv[optind], parameters.particleCount, "PARTICLES");
        }
        if (!parameters.sceneFile.empty()) {
            parameters.objects = readScene(parameters.sceneFile);
            std::uint64_t count = 0;
            double totalMass = 0;
            for (const SceneObject& object : parameters.objects) {
                count += object.particleCount;
                totalMass += object.mass;
            }
            if (count > std::numeric_limits<std::int32_t>::max() || !std::isfinite(totalMass)) {
                throw std::runtime_error("Scene particle count or total mass exceeds the supported range");
            }
            if (optind < argc && parameters.particleCount != count) {
                throw std::runtime_error("Positional particle count disagrees with scene");
            }
            parameters.particleCount = count;
        }
        if (parameters.prepareOnly) {
            parameters.stepCount = 0;
            if (parameters.frameCount > 0) parameters.frameCount = 1;
        }
        if (parameters.frameCount > 0 &&
            ((parameters.stepCount == 0 && parameters.frameCount != 1) ||
             (parameters.stepCount > 0 && (parameters.frameCount < 2 ||
               static_cast<std::int64_t>(parameters.frameCount) > static_cast<std::int64_t>(parameters.stepCount) + 1)))) {
            throw std::invalid_argument("--frames must include initial/final states and cannot exceed steps+1");
        }
        if (parameters.particleCount > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::length_error("Particle count exceeds the 32-bit index range");
        }
        if (parameters.particleCount == 0 || parameters.stepCount < 0 ||
            !std::isfinite(parameters.timestep) || parameters.timestep <= 0 ||
            parameters.snapshotInterval <= 0 || parameters.maxDepth < 0 ||
            !std::isfinite(parameters.theta) || parameters.theta < 0 ||
            !std::isfinite(parameters.softeningLength) || parameters.softeningLength <= 0 ||
            parameters.threadCount < 1 || parameters.outputDirectory.empty()) {
            throw std::invalid_argument("Invalid simulation parameter range or empty output path; use --help");
        }
        const auto totalStart = Clock::now();
        const auto parent = parameters.outputDirectory.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }
        if (!std::filesystem::create_directory(parameters.outputDirectory)) {
            throw std::runtime_error("Output directory already exists; choose a new run directory: " +
                                     parameters.outputDirectory.string());
        }

        std::vector<Particle> particles;
        std::vector<Node> nodes;
        std::vector<std::int32_t> nextParticle;
        std::vector<float> snapshotPositions(3 * parameters.particleCount);
        std::vector<int> snapshotSteps;
        std::cout << "Initializing particles...\n" << std::flush;
        double initialVelocityScale = 1.0;
        if (!parameters.initialStateFile.empty()) {
            readInitialState(particles, parameters.initialStateFile);
            if ((!parameters.objects.empty() || optind < argc) && particles.size() != parameters.particleCount) {
                throw std::runtime_error("Prepared state particle count disagrees with configuration");
            }
            parameters.particleCount = particles.size();
        } else if (!parameters.objects.empty()) {
            initializeScene(particles, parameters);
        } else {
            initializeParticles(particles, parameters);
            initialVelocityScale = normalizeInitialVirial(particles, nodes, nextParticle, parameters);
        }
        if (parameters.prepareOnly) writeInitialState(particles, parameters.outputDirectory / "initial_state.bin");

        double treeSeconds = 0.0;
        double aggregationSeconds = 0.0;
        double forceSeconds = 0.0;
        double integrationSeconds = 0.0;
        double outputSeconds = 0.0;
        auto saveFrame = [&](int completedStep) {
            const auto start = Clock::now();
            writeSnapshot(particles, parameters.outputDirectory / frameFilename(snapshotSteps.size()),
                          snapshotPositions);
            snapshotSteps.push_back(completedStep);
            outputSeconds += secondsSince(start);
        };

        std::cout << "Simulating " << parameters.particleCount << " particles with "
                  << parameters.threadCount << " requested threads\n"
                  << "Output: " << parameters.outputDirectory.string() << '\n' << std::flush;
        saveFrame(0);
        auto snapshotDue = [&](int step) {
            if (parameters.frameCount == 0) {
                return step % parameters.snapshotInterval == 0 || step == parameters.stepCount;
            }
            if (parameters.frameCount <= 1 || snapshotSteps.size() >= static_cast<std::size_t>(parameters.frameCount)) {
                return false;
            }
            // Round each requested output time to its nearest integration step.
            // N <= steps+1 guarantees distinct steps; uint64 avoids overflow.
            const std::uint64_t intervals = parameters.frameCount - 1;
            const std::uint64_t target = (snapshotSteps.size() * static_cast<std::uint64_t>(parameters.stepCount) +
                                           intervals / 2) / intervals;
            return static_cast<std::uint64_t>(step) == target;
        };
        auto updateAccelerations = [&]() {
            auto start = Clock::now();
            buildOctree(particles, nodes, nextParticle, parameters.maxDepth);
            treeSeconds += secondsSince(start);

            start = Clock::now();
            aggregateMass(0, particles, nodes, nextParticle);
            aggregationSeconds += secondsSince(start);

            start = Clock::now();
            calculateAccelerations(particles, nodes, nextParticle,
                                   parameters.theta, parameters.softeningLength, parameters.threadCount);
            forceSeconds += secondsSince(start);
        };
        const bool leapfrog = parameters.integrator == "leapfrog";
        // A prepared state does not contain accelerations. Always calculate the
        // initial force before the first half-kick, including loaded states.
        if (leapfrog && parameters.stepCount > 0) updateAccelerations();
        const int progressInterval = std::max(1, parameters.stepCount / 100);
        for (int step = 1; step <= parameters.stepCount; ++step) {
            if (leapfrog) {
                const auto start = Clock::now();
                leapfrogKickDrift(particles, parameters.timestep);
                integrationSeconds += secondsSince(start);
            }
            updateAccelerations();

            // The OpenMP force loop has completed before any positions change.
            const auto start = Clock::now();
            if (leapfrog) leapfrogFinishKick(particles, parameters.timestep);
            else integrateEuler(particles, parameters.timestep);
            integrationSeconds += secondsSince(start);

            if (snapshotDue(step)) {
                saveFrame(step);
            }
            if (step % progressInterval == 0 || step == parameters.stepCount) {
                std::cout << "Step " << step << '/' << parameters.stepCount
                          << ", time " << step * parameters.timestep << '\n' << std::flush;
            }
        }
        const auto metadataStart = Clock::now();
        writeMetadata(parameters, snapshotSteps, initialVelocityScale);
        outputSeconds += secondsSince(metadataStart);
        const double totalSeconds = secondsSince(totalStart);
        std::cout << std::fixed << std::setprecision(3)
                  << "Saved " << snapshotSteps.size() << " snapshots\n"
                  << "Tree construction: " << treeSeconds << " s\n"
                  << "Mass aggregation: " << aggregationSeconds << " s\n"
                  << "Force calculation: " << forceSeconds << " s\n"
                  << "Integration: " << integrationSeconds << " s\n"
                  << "Snapshot/metadata output: " << outputSeconds << " s\n"
                  << "Total run: " << totalSeconds << " s\n";
    } catch (const std::exception& error) {
        std::cerr << "Simulation failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
