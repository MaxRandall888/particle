#pragma once

#include "octree.hpp"
#include "particle.hpp"

#include <cstdint>
#include <vector>

// Accumulates into one target. Requires positive softening and aggregated masses.
void calculateAcceleration(std::int32_t particleIndex, std::int32_t nodeIndex,
                           const std::vector<Node>& nodes,
                           const std::vector<std::int32_t>& nextParticle,
                           std::vector<Particle>& particles,
                           double theta, double softeningLength);

void calculateAccelerations(std::vector<Particle>& particles,
                            const std::vector<Node>& nodes,
                            const std::vector<std::int32_t>& nextParticle,
                            double theta, double softeningLength, int threadCount);

void integrateEuler(std::vector<Particle>& particles, double timestep);

// Requires acceleration at the current positions; leaves velocities at half-step.
void leapfrogKickDrift(std::vector<Particle>& particles, double timestep);
// Requires acceleration recalculated at the drifted positions.
void leapfrogFinishKick(std::vector<Particle>& particles, double timestep);
