#pragma once

#include "particle.hpp"

#include <cstdint>
#include <vector>

struct Node {
    double center[3]{};
    double halfWidth{};

    double com[3]{};
    double mass{};

    std::int32_t children[8]{-1, -1, -1, -1, -1, -1, -1, -1};
    std::int32_t particleIndex = -1;
};

bool isLeaf(const Node& node);

void calculateBoundingCube(const std::vector<Particle>& particles, Node& root);

void buildOctree(const std::vector<Particle>& particles, std::vector<Node>& nodes,
                 std::vector<std::int32_t>& nextParticle, int maxDepth = 30);

void insertParticle(std::int32_t particleIndex, std::int32_t nodeIndex,
                    const std::vector<Particle>& particles, std::vector<Node>& nodes,
                    std::vector<std::int32_t>& nextParticle,
                    int depth = 0, int maxDepth = 30);

// Return the selected child, creating only that child if it is missing.
// Octant bits: x >= center sets bit 0, y sets bit 1, z sets bit 2.
std::int32_t subdivideNode(std::int32_t nodeIndex, int octant, std::vector<Node>& nodes);

void aggregateMass(std::int32_t nodeIndex, const std::vector<Particle>& particles,
                   std::vector<Node>& nodes,
                   const std::vector<std::int32_t>& nextParticle);
