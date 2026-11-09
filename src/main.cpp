#include "octree.hpp"
#include "particle.hpp"
#include "physics.hpp"

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
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <getopt.h>

struct SimulationParameters {
    std::size_t particleCount = 1000;
    int stepCount = 2000;
    double timestep = 0.001;
    double theta = 0.5;
    double softeningLength = 0.02;
    int threadCount = 16;
    int snapshotInterval = 10;
    int maxDepth = 30;
    std::uint32_t seed = 42;
    double lengthScaleKpc = 10.0;
    double massScaleSolarMasses = 1e11;
    std::filesystem::path outputDirectory = "output/run_001";
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

void initializeParticles(std::vector<Particle>& particles,
                         std::size_t particleCount, std::uint32_t seed) {
    particles.assign(particleCount, Particle{});
    if (particleCount == 0) {
        return;
    }

    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> coordinate(-1.0, 1.0);
    const double mass = 1.0 / static_cast<double>(particleCount);

    // Uniform cube, initially at rest, with total mass normalized to one.
    for (Particle& particle : particles) {
        particle.x = coordinate(generator);
        particle.y = coordinate(generator);
        particle.z = coordinate(generator);
        particle.mass = mass;
    }
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

void writeMetadata(const SimulationParameters& parameters, const std::vector<int>& snapshotSteps) {
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
         << "  \"max_tree_depth\": " << parameters.maxDepth << ",\n"
         << "  \"seed\": " << parameters.seed << ",\n"
         << "  \"integrator\": \"explicit_euler\",\n"
         << "  \"softening_model\": \"plummer\",\n"
         << "  \"initial_conditions\": {\"distribution\": \"uniform_cube\", "
            "\"coordinate_min\": -1, \"coordinate_max\": 1, "
            "\"total_mass\": 1, \"initial_velocity\": [0, 0, 0]},\n"
         << "  \"units\": {\n"
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
            SnapshotInterval = 1000, Timestep, OpeningAngle, Softening, Seed
        };
        const option options[]{
            {"threads", required_argument, nullptr, 't'},
            {"steps", required_argument, nullptr, 's'},
            {"snapshot-interval", required_argument, nullptr, SnapshotInterval},
            {"dt", required_argument, nullptr, Timestep},
            {"theta", required_argument, nullptr, OpeningAngle},
            {"softening", required_argument, nullptr, Softening},
            {"seed", required_argument, nullptr, Seed},
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
            case Timestep:
                parseNumber(optarg, parameters.timestep, "--dt");
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
                          << "  --dt VALUE             Positive timestep in simulation units (default: "
                          << defaults.timestep << ")\n"
                          << "  --theta VALUE          Nonnegative opening angle; 0 disables approximation (default: "
                          << defaults.theta << ")\n"
                          << "  --softening VALUE      Positive Plummer softening length (default: "
                          << defaults.softeningLength << ")\n"
                          << "  --seed N               Unsigned 32-bit random seed (default: "
                          << defaults.seed << ")\n"
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
        initializeParticles(particles, parameters.particleCount, parameters.seed);

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
                  << "Output: " << parameters.outputDirectory.string() << '\n';
        saveFrame(0);
        const int progressInterval = parameters.stepCount / 10 > 0 ? parameters.stepCount / 10 : 1;
        for (int step = 1; step <= parameters.stepCount; ++step) {
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

            // The OpenMP force loop has completed before any positions change.
            start = Clock::now();
            integrateEuler(particles, parameters.timestep);
            integrationSeconds += secondsSince(start);

            if (step % parameters.snapshotInterval == 0 || step == parameters.stepCount) {
                saveFrame(step);
            }
            if (step % progressInterval == 0 || step == parameters.stepCount) {
                std::cout << "Step " << step << '/' << parameters.stepCount
                          << ", time " << step * parameters.timestep << '\n';
            }
        }
        const auto metadataStart = Clock::now();
        writeMetadata(parameters, snapshotSteps);
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
