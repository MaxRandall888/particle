#include "octree.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace {

int chooseOctant(const Particle& particle, const Node& node) {
    return (particle.x >= node.center[0] ? 1 : 0)
         | (particle.y >= node.center[1] ? 2 : 0)
         | (particle.z >= node.center[2] ? 4 : 0);
}

bool canSubdivide(const Node& node) {
    const double childHalfWidth = node.halfWidth / 2.0;
    if (childHalfWidth <= 0.0) {
        return false;
    }
    for (double center : node.center) {
        const double lowerCenter = center - childHalfWidth;
        const double upperCenter = center + childHalfWidth;
        if (!std::isfinite(lowerCenter) || !std::isfinite(upperCenter) ||
            lowerCenter >= center || upperCenter <= center) {
            return false;
        }
    }
    return true;
}

} // namespace

bool isLeaf(const Node& node) {
    return std::all_of(std::begin(node.children), std::end(node.children),
                       [](std::int32_t child) { return child == -1; });
}

void calculateBoundingCube(const std::vector<Particle>& particles, Node& root) {
    root = Node{};
    if (particles.empty()) {
        return;
    }

    double minimum[3]{particles.front().x, particles.front().y, particles.front().z};
    double maximum[3]{minimum[0], minimum[1], minimum[2]};
    for (const Particle& particle : particles) {
        const double position[3]{particle.x, particle.y, particle.z};
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(position[axis])) {
                throw std::invalid_argument("Particle coordinates must be finite");
            }
            minimum[axis] = std::min(minimum[axis], position[axis]);
            maximum[axis] = std::max(maximum[axis], position[axis]);
        }
    }

    for (int axis = 0; axis < 3; ++axis) {
        root.center[axis] = std::midpoint(minimum[axis], maximum[axis]);
        root.halfWidth = std::max({root.halfWidth,
                                  maximum[axis] - root.center[axis],
                                  root.center[axis] - minimum[axis]});
    }
    // Round outward. Give a point distribution a positive, representable cube.
    if (root.halfWidth == 0.0) {
        const double scale = std::max({1.0, std::abs(root.center[0]),
                                      std::abs(root.center[1]), std::abs(root.center[2])});
        root.halfWidth = scale * std::numeric_limits<double>::epsilon();
    }
    root.halfWidth = std::nextafter(root.halfWidth, std::numeric_limits<double>::infinity());
    if (!std::isfinite(root.halfWidth)) {
        throw std::overflow_error("Bounding cube exceeds the supported coordinate range");
    }
}

void buildOctree(const std::vector<Particle>& particles, std::vector<Node>& nodes,
                 std::vector<std::int32_t>& nextParticle, int maxDepth) {
    if (maxDepth < 0) {
        throw std::invalid_argument("Maximum tree depth must be nonnegative");
    }
    if (particles.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::length_error("Particle count exceeds the index range");
    }

    nodes.clear();
    nextParticle.assign(particles.size(), -1);
    if (particles.empty()) {
        return;
    }
    nodes.reserve(2 * particles.size());
    Node root;
    calculateBoundingCube(particles, root);
    nodes.push_back(root);

    for (std::size_t p = 0; p < particles.size(); ++p) {
        insertParticle(static_cast<std::int32_t>(p), 0, particles, nodes,
                       nextParticle, 0, maxDepth);
    }
}

void insertParticle(std::int32_t particleIndex, std::int32_t nodeIndex,
                    const std::vector<Particle>& particles, std::vector<Node>& nodes,
                    std::vector<std::int32_t>& nextParticle, int depth, int maxDepth) {
    if (isLeaf(nodes[nodeIndex])) {
        const std::int32_t first = nodes[nodeIndex].particleIndex;
        if (first == -1) {
            nodes[nodeIndex].particleIndex = particleIndex;
            return;
        }
        if (nextParticle[first] != -1 || depth >= maxDepth || !canSubdivide(nodes[nodeIndex])) {
            nextParticle[particleIndex] = first;
            nodes[nodeIndex].particleIndex = particleIndex;
            return;
        }

        // The existing singleton and the incoming particle descend separately.
        nodes[nodeIndex].particleIndex = -1;
        const int oldOctant = chooseOctant(particles[first], nodes[nodeIndex]);
        const std::int32_t oldChild = subdivideNode(nodeIndex, oldOctant, nodes);
        insertParticle(first, oldChild, particles, nodes, nextParticle, depth + 1, maxDepth);
    }

    const int octant = chooseOctant(particles[particleIndex], nodes[nodeIndex]);
    const std::int32_t child = subdivideNode(nodeIndex, octant, nodes);
    insertParticle(particleIndex, child, particles, nodes, nextParticle, depth + 1, maxDepth);
}

std::int32_t subdivideNode(std::int32_t nodeIndex, int octant, std::vector<Node>& nodes) {
    if (nodes[nodeIndex].children[octant] != -1) {
        return nodes[nodeIndex].children[octant];
    }
    if (nodes.size() >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::length_error("Node count exceeds the index range");
    }

    Node child;
    child.halfWidth = nodes[nodeIndex].halfWidth / 2.0;
    for (int axis = 0; axis < 3; ++axis) {
        const double direction = (octant & (1 << axis)) ? 1.0 : -1.0;
        child.center[axis] = nodes[nodeIndex].center[axis] + direction * child.halfWidth;
    }
    const auto childIndex = static_cast<std::int32_t>(nodes.size());
    nodes.push_back(child);
    // Reacquire by index after push_back, which can invalidate references.
    nodes[nodeIndex].children[octant] = childIndex;
    return childIndex;
}

void aggregateMass(std::int32_t nodeIndex, const std::vector<Particle>& particles,
                   std::vector<Node>& nodes,
                   const std::vector<std::int32_t>& nextParticle) {
    Node& node = nodes[nodeIndex];
    node.mass = 0.0;
    std::fill(std::begin(node.com), std::end(node.com), 0.0);

    if (isLeaf(node)) {
        for (std::int32_t p = node.particleIndex; p != -1; p = nextParticle[p]) {
            const Particle& particle = particles[p];
            node.mass += particle.mass;
            node.com[0] += particle.mass * particle.x;
            node.com[1] += particle.mass * particle.y;
            node.com[2] += particle.mass * particle.z;
        }
    } else {
        for (std::int32_t childIndex : node.children) {
            if (childIndex == -1) {
                continue;
            }
            aggregateMass(childIndex, particles, nodes, nextParticle);
            const Node& child = nodes[childIndex];
            node.mass += child.mass;
            for (int axis = 0; axis < 3; ++axis) {
                node.com[axis] += child.mass * child.com[axis];
            }
        }
    }
    if (node.mass != 0.0) {
        for (double& coordinate : node.com) {
            coordinate /= node.mass;
        }
    }
}
