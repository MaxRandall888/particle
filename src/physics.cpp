#include "physics.hpp"

#include <cmath>
#include <stdexcept>

namespace {

void addAcceleration(Particle& target, double x, double y, double z,
                     double mass, double softeningSquared) {
    const double dx = x - target.x;
    const double dy = y - target.y;
    const double dz = z - target.z;
    const double distanceSquared = dx * dx + dy * dy + dz * dz + softeningSquared;
    const double inverseDistance = 1.0 / std::sqrt(distanceSquared);
    const double factor = mass * inverseDistance * inverseDistance * inverseDistance;
    target.ax += factor * dx;
    target.ay += factor * dy;
    target.az += factor * dz;
}

void traverse(std::int32_t particleIndex, std::int32_t nodeIndex,
              const std::vector<Node>& nodes,
              const std::vector<std::int32_t>& nextParticle,
              std::vector<Particle>& particles, double thetaSquared,
              double softeningSquared, bool containsTarget) {
    const Node& node = nodes[nodeIndex];
    Particle& target = particles[particleIndex];
    if (isLeaf(node)) {
        for (std::int32_t p = node.particleIndex; p != -1; p = nextParticle[p]) {
            if (p != particleIndex) {
                const Particle& source = particles[p];
                addAcceleration(target, source.x, source.y, source.z,
                                source.mass, softeningSquared);
            }
        }
        return;
    }

    const double dx = node.com[0] - target.x;
    const double dy = node.com[1] - target.y;
    const double dz = node.com[2] - target.z;
    const double distanceSquared = dx * dx + dy * dy + dz * dz;
    const double sideLength = 2.0 * node.halfWidth;
    if (!containsTarget && sideLength * sideLength < thetaSquared * distanceSquared) {
        addAcceleration(target, node.com[0], node.com[1], node.com[2],
                        node.mass, softeningSquared);
        return;
    }

    // Follow the same split-plane rule used during insertion. The target's
    // ancestor branch must always be opened, irrespective of the opening angle.
    const int targetOctant = (target.x >= node.center[0] ? 1 : 0)
                           | (target.y >= node.center[1] ? 2 : 0)
                           | (target.z >= node.center[2] ? 4 : 0);
    for (int octant = 0; octant < 8; ++octant) {
        const std::int32_t child = node.children[octant];
        if (child != -1) {
            traverse(particleIndex, child, nodes, nextParticle, particles,
                     thetaSquared, softeningSquared,
                     containsTarget && octant == targetOctant);
        }
    }
}

} // namespace

void calculateAcceleration(std::int32_t particleIndex, std::int32_t nodeIndex,
                           const std::vector<Node>& nodes,
                           const std::vector<std::int32_t>& nextParticle,
                           std::vector<Particle>& particles,
                           double theta, double softeningLength) {
    // Accumulate into the target's existing acceleration. Start conservatively
    // on a containing branch; normal full-tree traversal starts at node zero.
    traverse(particleIndex, nodeIndex, nodes, nextParticle, particles,
             theta * theta, softeningLength * softeningLength, true);
}

void calculateAccelerations(std::vector<Particle>& particles,
                            const std::vector<Node>& nodes,
                            const std::vector<std::int32_t>& nextParticle,
                            double theta, double softeningLength, int threadCount) {
    if (!std::isfinite(theta) || theta < 0.0 ||
        !std::isfinite(softeningLength) || softeningLength <= 0.0 || threadCount < 1) {
        throw std::invalid_argument("Require theta >= 0, softening > 0, and threadCount >= 1");
    }
    if (nextParticle.size() != particles.size() || (!particles.empty() && nodes.empty())) {
        throw std::invalid_argument("Build the tree and particle links before calculating forces");
    }

    // Positions, masses, nodes, and links stay read-only throughout this loop.
    #pragma omp parallel for schedule(static) num_threads(threadCount)
    for (std::size_t p = 0; p < particles.size(); ++p) {
        particles[p].ax = 0.0;
        particles[p].ay = 0.0;
        particles[p].az = 0.0;
        calculateAcceleration(static_cast<std::int32_t>(p), 0, nodes, nextParticle,
                              particles, theta, softeningLength);
    }
}

void integrateEuler(std::vector<Particle>& particles, double timestep) {
    for (Particle& particle : particles) {
        // Explicit Euler uses the old velocity for the position update.
        particle.x += particle.vx * timestep;
        particle.y += particle.vy * timestep;
        particle.z += particle.vz * timestep;

        particle.vx += particle.ax * timestep;
        particle.vy += particle.ay * timestep;
        particle.vz += particle.az * timestep;
    }
}

void leapfrogKickDrift(std::vector<Particle>& particles, double timestep) {
    for (Particle& particle : particles) {
        particle.vx += particle.ax * (0.5 * timestep);
        particle.vy += particle.ay * (0.5 * timestep);
        particle.vz += particle.az * (0.5 * timestep);
        particle.x += particle.vx * timestep;
        particle.y += particle.vy * timestep;
        particle.z += particle.vz * timestep;
    }
}

void leapfrogFinishKick(std::vector<Particle>& particles, double timestep) {
    for (Particle& particle : particles) {
        particle.vx += particle.ax * (0.5 * timestep);
        particle.vy += particle.ay * (0.5 * timestep);
        particle.vz += particle.az * (0.5 * timestep);
    }
}
