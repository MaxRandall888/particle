#include "octree.hpp"
#include "physics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double actual, double expected, const std::string& message) {
    const double tolerance = 1e-12 + 1e-11 * std::abs(expected);
    expect(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
           message + ": expected " + std::to_string(expected) +
           ", got " + std::to_string(actual));
}

Particle particle(double x, double y, double z, double mass = 1.0) {
    Particle p{};
    p.x = x;
    p.y = y;
    p.z = z;
    p.mass = mass;
    return p;
}

// Check reachability and uniqueness without assuming the node allocation order.
// Return the largest leaf population; bounded walks detect accidental cycles.
int checkTree(const std::vector<Node>& nodes,
              const std::vector<std::int32_t>& nextParticle,
              std::size_t particleCount, int depthLimit) {
    expect(nextParticle.size() == particleCount, "Link array size");
    std::vector<int> seenNodes(nodes.size());
    std::vector<int> seenParticles(particleCount);
    int largestBucket = 0;
    std::function<void(std::int32_t, int)> visit = [&](std::int32_t index, int depth) {
        expect(index >= 0 && static_cast<std::size_t>(index) < nodes.size(), "Valid node index");
        expect(++seenNodes[index] == 1, "Node has exactly one parent; no cycles");
        expect(depth <= depthLimit, "Tree respects depth limit");
        const Node& node = nodes[index];
        if (isLeaf(node)) {
            int count = 0;
            for (std::int32_t p = node.particleIndex; p != -1; p = nextParticle[p]) {
                expect(p >= 0 && static_cast<std::size_t>(p) < particleCount, "Valid particle index");
                expect(++seenParticles[p] == 1, "Particle appears once; no chain cycles");
                ++count;
            }
            largestBucket = std::max(largestBucket, count);
        } else {
            expect(node.particleIndex == -1, "Internal node has no leaf particle");
            for (std::int32_t child : node.children) {
                if (child != -1) {
                    visit(child, depth + 1);
                }
            }
        }
    };
    if (!nodes.empty()) {
        visit(0, 0);
    }
    for (int count : seenNodes) expect(count == 1, "Every allocated node is reachable");
    for (int count : seenParticles) expect(count == 1, "Every particle is present");
    return largestBucket;
}

std::vector<std::array<double, 3>> directAccelerations(
    const std::vector<Particle>& particles, double softening) {
    std::vector<std::array<double, 3>> result(particles.size());
    for (std::size_t i = 0; i < particles.size(); ++i) {
        for (std::size_t j = 0; j < particles.size(); ++j) {
            if (i == j) continue;
            const double dx = particles[j].x - particles[i].x;
            const double dy = particles[j].y - particles[i].y;
            const double dz = particles[j].z - particles[i].z;
            const double radius = std::sqrt(dx * dx + dy * dy + dz * dz + softening * softening);
            const double factor = particles[j].mass / (radius * radius * radius);
            result[i][0] += factor * dx;
            result[i][1] += factor * dy;
            result[i][2] += factor * dz;
        }
    }
    return result;
}

void testEmpty() {
    std::vector<Particle> particles;
    std::vector<Node> nodes;
    std::vector<std::int32_t> links;
    buildOctree(particles, nodes, links);
    expect(nodes.empty() && links.empty(), "Empty input produces an empty tree");
    calculateAccelerations(particles, nodes, links, 0.5, 0.1, 1);
    integrateEuler(particles, 0.1);
}

void testBoundingCube() {
    const std::vector<Particle> particles{particle(-8, -4, -2), particle(-2, 6, 4)};
    Node root;
    calculateBoundingCube(particles, root);
    near(root.center[0], -5, "Root center x");
    near(root.center[1], 1, "Root center y");
    near(root.center[2], 1, "Root center z");
    near(root.halfWidth, 5, "Largest extent determines half-width");
    expect(root.halfWidth > 5, "Root includes an outward margin");
    for (const Particle& p : particles) {
        expect(std::abs(p.x - root.center[0]) <= root.halfWidth &&
               std::abs(p.y - root.center[1]) <= root.halfWidth &&
               std::abs(p.z - root.center[2]) <= root.halfWidth, "Root contains both extremes");
    }
}

void testSingleton() {
    std::vector<Particle> particles{particle(-3, 2, 5, 7)};
    particles[0].ax = particles[0].ay = particles[0].az = 123;
    std::vector<Node> nodes;
    std::vector<std::int32_t> links;
    buildOctree(particles, nodes, links);
    expect(nodes.size() == 1 && isLeaf(nodes[0]), "Singleton has one leaf");
    expect(nodes[0].particleIndex == 0 && links[0] == -1, "Singleton chain terminates");
    expect(nodes[0].halfWidth > 0, "Point distribution has a positive root width");
    aggregateMass(0, particles, nodes, links);
    near(nodes[0].mass, 7, "Singleton mass");
    near(nodes[0].com[0], -3, "Singleton center of mass x");
    near(nodes[0].com[1], 2, "Singleton center of mass y");
    near(nodes[0].com[2], 5, "Singleton center of mass z");
    calculateAccelerations(particles, nodes, links, 1000, 0.1, 1);
    near(particles[0].ax, 0, "No self-gravity x; previous acceleration cleared");
    near(particles[0].ay, 0, "No self-gravity y");
    near(particles[0].az, 0, "No self-gravity z");
}

void testOctants() {
    std::vector<Particle> particles;
    for (int z : {-1, 1}) for (int y : {-1, 1}) for (int x : {-1, 1}) {
        particles.push_back(particle(x, y, z));
    }
    std::vector<Node> nodes;
    std::vector<std::int32_t> links;
    buildOctree(particles, nodes, links);
    expect(nodes.size() == 9, "Eight octants require only root plus eight leaves");
    expect(checkTree(nodes, links, particles.size(), 1) == 1, "Each octant holds one particle");
    for (int octant = 0; octant < 8; ++octant) {
        const Node& child = nodes[nodes[0].children[octant]];
        expect(child.particleIndex == octant, "Documented octant numbering");
        near(child.halfWidth, nodes[0].halfWidth / 2, "Child half-width");
        near(child.center[0], particles[octant].x * child.halfWidth, "Child center x");
        near(child.center[1], particles[octant].y * child.halfWidth, "Child center y");
        near(child.center[2], particles[octant].z * child.halfWidth, "Child center z");
    }
}

void testDeepSubdivision() {
    const std::vector<Particle> particles{
        particle(0, 0, 0), particle(1e-12, 1e-12, 1e-12), particle(1, 1, 1)};
    std::vector<Node> nodes;
    std::vector<std::int32_t> links;
    buildOctree(particles, nodes, links, 30);
    expect(nodes.size() > 2 * particles.size(), "Close pair grows beyond initial capacity estimate");
    expect(checkTree(nodes, links, particles.size(), 30) == 2, "Close pair retained in terminal bucket");
}

void testCoincident() {
    std::vector<Particle> particles(5, particle(2, -3, 4));
    particles.push_back(particle(10, 10, 10));
    std::vector<Node> nodes;
    std::vector<std::int32_t> links;
    buildOctree(particles, nodes, links, 4);
    expect(checkTree(nodes, links, particles.size(), 4) == 5, "All coincident particles retained");
    aggregateMass(0, particles, nodes, links);
    calculateAccelerations(particles, nodes, links, 0, 0.1, 1);
    const auto expected = directAccelerations(particles, 0.1);
    for (std::size_t p = 0; p < particles.size(); ++p) {
        near(particles[p].ax, expected[p][0], "Coincident force x");
        near(particles[p].ay, expected[p][1], "Coincident force y");
        near(particles[p].az, expected[p][2], "Coincident force z");
    }
}

void testMass() {
    const std::vector<Particle> particles{particle(0, -2, 4, 1), particle(4, 2, 0, 3)};
    for (int limit : {0, 30}) {
        std::vector<Node> nodes;
        std::vector<std::int32_t> links;
        buildOctree(particles, nodes, links, limit);
        checkTree(nodes, links, particles.size(), limit);
        // Repeated aggregation must replace, rather than accumulate, old totals.
        for (int repeat = 0; repeat < 2; ++repeat) {
            aggregateMass(0, particles, nodes, links);
            near(nodes[0].mass, 4, "Total mass in bucket or internal node");
            near(nodes[0].com[0], 3, "Mass-weighted center x");
            near(nodes[0].com[1], 1, "Mass-weighted center y");
            near(nodes[0].com[2], 1, "Mass-weighted center z");
        }
    }
}

void testRebuild() {
    std::vector<Particle> particles{particle(-1, 0, 0), particle(1, 0, 0), particle(0, 1, 0)};
    std::vector<Node> nodes;
    std::vector<std::int32_t> links;
    buildOctree(particles, nodes, links, 0);
    expect(checkTree(nodes, links, particles.size(), 0) == 3, "Forced root bucket");
    buildOctree(particles, nodes, links);
    expect(checkTree(nodes, links, particles.size(), 30) == 1, "Rebuild clears old bucket links");
    const std::size_t capacity = nodes.capacity();
    particles.resize(1);
    buildOctree(particles, nodes, links);
    checkTree(nodes, links, 1, 30);
    expect(nodes.capacity() == capacity, "Rebuild retains node capacity");
    particles.clear();
    buildOctree(particles, nodes, links);
    expect(nodes.empty() && links.empty(), "Rebuild to empty clears prior state");
}

void testTwoBody() {
    std::vector<Particle> particles{particle(-1, 0, 0, 2), particle(1, 0, 0, 3)};
    std::vector<Node> nodes;
    std::vector<std::int32_t> links;
    buildOctree(particles, nodes, links);
    aggregateMass(0, particles, nodes, links);
    const double denominator = std::pow(4.0 + 0.25, 1.5);
    // Even a huge angle must not approximate a node containing the target.
    calculateAccelerations(particles, nodes, links, 1e6, 0.5, 1);
    near(particles[0].ax, 6 / denominator, "Left particle accelerates right");
    near(particles[1].ax, -4 / denominator, "Right particle accelerates left");
    for (const Particle& p : particles) {
        near(p.ay, 0, "Two-body y acceleration");
        near(p.az, 0, "Two-body z acceleration");
    }
    near(2 * particles[0].ax + 3 * particles[1].ax, 0, "Equal and opposite forces");
}

void testDistantApproximation() {
    std::vector<Particle> particles{
        particle(-10, -1, 0), particle(10, 1, 0, 2), particle(10, 1.1, 0, 3)};
    std::vector<Node> nodes;
    std::vector<std::int32_t> links;
    buildOctree(particles, nodes, links);
    aggregateMass(0, particles, nodes, links);
    calculateAccelerations(particles, nodes, links, 1, 0.1, 1);
    // The two distant sources form an internal node accepted at theta = 1.
    // Their combined mass is 5, at (10, 1.06, 0).
    const double denominator = std::pow(20 * 20 + 2.06 * 2.06 + 0.01, 1.5);
    near(particles[0].ax, 5 * 20 / denominator, "Distant node monopole x");
    near(particles[0].ay, 5 * 2.06 / denominator, "Distant node monopole y");
    near(particles[0].az, 0, "Distant node monopole z");
}

void testDirectAndParallel() {
    std::vector<Particle> particles;
    for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) for (int z = -1; z <= 1; ++z) {
        particles.push_back(particle(x, y, z, 1 + particles.size() % 3));
    }
    std::vector<Node> nodes;
    std::vector<std::int32_t> links;
    const auto expected = directAccelerations(particles, 0.1);
    // Check both ordinary leaves and a bucket containing all particles.
    for (int limit : {0, 30}) {
        buildOctree(particles, nodes, links, limit);
        checkTree(nodes, links, particles.size(), limit);
        aggregateMass(0, particles, nodes, links);
        calculateAccelerations(particles, nodes, links, 0, 0.1, 1);
        for (std::size_t p = 0; p < particles.size(); ++p) {
            near(particles[p].ax, expected[p][0], "Direct summation x");
            near(particles[p].ay, expected[p][1], "Direct summation y");
            near(particles[p].az, expected[p][2], "Direct summation z");
        }
        calculateAccelerations(particles, nodes, links, 0.5, 0.1, 1);
        const auto sequential = particles;
        calculateAccelerations(particles, nodes, links, 0.5, 0.1, 4);
        for (std::size_t p = 0; p < particles.size(); ++p) {
            near(particles[p].ax, sequential[p].ax, "Parallel agreement x");
            near(particles[p].ay, sequential[p].ay, "Parallel agreement y");
            near(particles[p].az, sequential[p].az, "Parallel agreement z");
            near(particles[p].x, sequential[p].x, "Force calculation preserves position x");
            near(particles[p].y, sequential[p].y, "Force calculation preserves position y");
            near(particles[p].z, sequential[p].z, "Force calculation preserves position z");
        }
    }
}

void testEuler() {
    Particle p = particle(1, -2, 3, 7);
    p.vx = 2; p.vy = -4; p.vz = 6;
    p.ax = 8; p.ay = -2; p.az = 4;
    std::vector<Particle> particles{p, particle(-5, 6, -7)};
    integrateEuler(particles, 0.5);
    near(particles[0].x, 2, "Euler x uses old velocity");
    near(particles[0].y, -4, "Euler y uses old velocity");
    near(particles[0].z, 6, "Euler z uses old velocity");
    near(particles[0].vx, 6, "Euler vx");
    near(particles[0].vy, -5, "Euler vy");
    near(particles[0].vz, 8, "Euler vz");
    near(particles[0].ax, 8, "Euler preserves acceleration x");
    near(particles[0].ay, -2, "Euler preserves acceleration y");
    near(particles[0].az, 4, "Euler preserves acceleration z");
    near(particles[0].mass, 7, "Euler preserves mass");
    near(particles[1].x, -5, "Stationary particle x");
    near(particles[1].y, 6, "Stationary particle y");
    near(particles[1].z, -7, "Stationary particle z");
    const auto before = particles;
    integrateEuler(particles, 0);
    for (std::size_t i = 0; i < particles.size(); ++i) {
        near(particles[i].x, before[i].x, "Zero timestep x");
        near(particles[i].y, before[i].y, "Zero timestep y");
        near(particles[i].z, before[i].z, "Zero timestep z");
        near(particles[i].vx, before[i].vx, "Zero timestep vx");
        near(particles[i].vy, before[i].vy, "Zero timestep vy");
        near(particles[i].vz, before[i].vz, "Zero timestep vz");
    }
}

} // namespace

int main() {
    const std::pair<const char*, void (*)()> tests[]{
        {"empty input", testEmpty},
        {"bounding cube", testBoundingCube},
        {"singleton and self-gravity", testSingleton},
        {"eight octants and child geometry", testOctants},
        {"deep subdivision and vector growth", testDeepSubdivision},
        {"coincident particle bucket", testCoincident},
        {"mass and center of mass", testMass},
        {"tree rebuild", testRebuild},
        {"softened two-body gravity", testTwoBody},
        {"distant node approximation", testDistantApproximation},
        {"direct summation and parallel agreement", testDirectAndParallel},
        {"explicit Euler", testEuler},
    };
    int failures = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "PASS: " << name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL: " << name << ": " << error.what() << '\n';
        }
    }
    std::cout << std::size(tests) - failures << '/' << std::size(tests) << " tests passed\n";
    return failures == 0 ? 0 : 1;
}
