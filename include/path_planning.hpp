#pragma once

#include <array>
#include <deque>
#include <memory_resource>
#include <queue>
#include <stdexcept>


class path_planning {
public:
  inline static constexpr std::size_t map_width = 3;
  inline static constexpr std::size_t map_height = 6;

  enum class kfs_type { empty = 0, r1kfs, r2kfs, falsekfs };

  struct point {
    int x;
    int y;
  };

  path_planning() = default;
  ~path_planning() = default;

  void set_kfs_type(const point &p, kfs_type type) {
    if (p.x < 0 || p.x >= map_width || p.y < 0 || p.y >= map_height) {
      throw std::out_of_range("Point is out of map bounds");
    }
    m_map[p.x][p.y] = type;
  }

  kfs_type get_kfs_type(const point &p) const {
    if (p.x < 0 || p.x >= map_width || p.y < 0 || p.y >= map_height) {
      throw std::out_of_range("Point is out of map bounds");
    }
    return m_map[p.x][p.y];
  }

  struct path_node {
    point p;
    kfs_type type;
  };

  struct a_star_node {
    point p;
    kfs_type type;
    std::size_t g_cost; // Cost from start to current node
    std::size_t h_cost; // Heuristic cost from current node to end
    std::array<std::array<bool, map_height>, map_width>
        walked{}; // To track walked nodes
    std::size_t f_cost() const { return g_cost + h_cost; } // Total cost
  };

  std::queue<a_star_node> find_path(const point &start,
                                    const point &end) const {
    std::queue<a_star_node> result;
    auto path = a_star(start, end);
    if (path.empty()) {
      return {};
    }

    while (!path.empty()) {
      result.push(std::move(path.front()));
      path.pop();
    }
    return result;
  }

protected:
  static std::size_t get_manhattan_distance(const point &a, const point &b) {
    return std::abs(a.x - b.x) + std::abs(a.y - b.y);
  }

  struct compare_a_star_node {
    using queue_t = std::queue<
        a_star_node,
        std::deque<a_star_node, std::pmr::polymorphic_allocator<a_star_node>>>;
    bool operator()(const queue_t &a, const queue_t &b) const {
      return a.back().f_cost() > b.back().f_cost();
    }
  };

  std::queue<
      a_star_node,
      std::deque<a_star_node, std::pmr::polymorphic_allocator<a_star_node>>>
  a_star(const point &start, const point &end) const {
    using queue_t = std::queue<
        a_star_node,
        std::deque<a_star_node, std::pmr::polymorphic_allocator<a_star_node>>>;
    std::priority_queue<queue_t, std::pmr::vector<queue_t>, compare_a_star_node>
        path{compare_a_star_node{}, &pool_resource};
    {
      // 初始化路径
      queue_t initial_path{&pool_resource};
      a_star_node start_node{start, get_kfs_type(start), 0,
                             get_manhattan_distance(start, end)};
      start_node.walked[start.x][start.y] = true;
      initial_path.push(std::move(start_node));
      path.push(std::move(initial_path));
    }

    while (!path.empty()) {
      queue_t current_path{&pool_resource};
      current_path = path.top();
      path.pop();
      const a_star_node current_node = current_path.back();
      if (current_node.p.x == end.x && current_node.p.y == end.y) {
        return current_path;
      }

      auto get_r2kfs_count = [this](const point &p) {
        std::size_t count = 0;
        for (int i = 0; i < 5; ++i) {
          point adjacent_point{p.x, p.y};
          switch (i) {
          case 0:
            adjacent_point.y -= 1;
            break; // Up
          case 1:
            adjacent_point.y += 1;
            break; // Down
          case 2:
            adjacent_point.x -= 1;
            break; // Left
          case 3:
            adjacent_point.x += 1;
            break; // Right
          case 4:
            break; // Current node itself
          default:
            break;
          }
          if (adjacent_point.x >= 0 && adjacent_point.x < map_width &&
              adjacent_point.y >= 0 && adjacent_point.y < map_height &&
              get_kfs_type(adjacent_point) == kfs_type::r2kfs) {
            ++count;
          }
        }
        return count;
      };

      for (int i = 0; i < 4; ++i) {
        auto generate_next_path = [&](point next_point) {
          if (current_node.walked[next_point.x][next_point.y])
            return; // Already walked
          const kfs_type next_type = get_kfs_type(next_point);
          if (next_type == kfs_type::falsekfs)
            return; // Obstacle
          std::size_t r2kfs_count = get_r2kfs_count(next_point);
          a_star_node next_node{
              next_point, next_type,
              current_node.g_cost + 1 + (r2kfs_count > 0 ? 0 : 1) +
                  (next_type == kfs_type::r1kfs ? 1 : 0),
              get_manhattan_distance(next_point, end), current_node.walked};
          next_node.walked[next_point.x][next_point.y] = true;
          queue_t new_path{&pool_resource};
          new_path = current_path;
          new_path.push(next_node);
          path.push(std::move(new_path));
        };

        switch (i) {
        case 0: // Up
        {
          point next_point{current_node.p.x, current_node.p.y - 1};
          if (next_point.y < 0)
            continue; // Out of bounds
          generate_next_path(next_point);
        } break;
        case 1: // Down
        {
          point next_point{current_node.p.x, current_node.p.y + 1};
          if (next_point.y >= map_height)
            continue; // Out of bounds
          generate_next_path(next_point);
        } break;
        case 2: // Left
        {
          point next_point{current_node.p.x - 1, current_node.p.y};
          if (next_point.x < 0)
            continue; // Out of bounds
          generate_next_path(next_point);
        } break;
        case 3: // Right
        {
          point next_point{current_node.p.x + 1, current_node.p.y};
          if (next_point.x >= map_width)
            continue; // Out of bounds
          generate_next_path(next_point);
        } break;
        }
      }
    }
    return {};
  }

private:
  std::array<std::array<kfs_type, map_height>, map_width> m_map{};

  inline static std::pmr::synchronized_pool_resource pool_resource{};
};
