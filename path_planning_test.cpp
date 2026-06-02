#include "path_planning.hpp"

#include <functional>
#include <iostream>
#include <fstream>
#include <unordered_set>

path_planning planner;
std::size_t test_count = 0;
std::ofstream fp("path_planning_test_output.txt");

struct hash {
  std::size_t operator()(const std::array<std::array<path_planning::kfs_type, path_planning::map_height>,
    path_planning::map_width>& map) const {
    std::size_t result = 0;
    for (int i = 0; i < path_planning::map_width; ++i) {
      for (int j = 0; j < path_planning::map_height; ++j) {
        switch (map[i][j])
        {
        case path_planning::kfs_type::empty:
          /* code */
          break;

        case path_planning::kfs_type::r1kfs:
          result += 100 * (i * path_planning::map_height + j);
          break;

        case path_planning::kfs_type::r2kfs:
          result += 10000 * (i * path_planning::map_height + j);
          break;

        case path_planning::kfs_type::falsekfs:
          result += 1000000 * (i * path_planning::map_height + j);
          break;

        default:
          break;
        }
      }
    }
    return result;
  }
};

int main() {
  std::streambuf* cout_buf = std::cout.rdbuf();
  std::cout.rdbuf(fp ? fp.rdbuf() : cout_buf); // Redirect std::cout to file if opened successfully

  // 设置障碍物和特殊节点
  std::function<void(int)> setup_map;
  std::vector<path_planning::kfs_type> types = {
    path_planning::kfs_type::r1kfs,
    path_planning::kfs_type::r1kfs,
    path_planning::kfs_type::r1kfs,
    path_planning::kfs_type::r2kfs,
    path_planning::kfs_type::r2kfs,
    path_planning::kfs_type::r2kfs,
    path_planning::kfs_type::r2kfs
  };
  std::unordered_set<std::array<std::array<path_planning::kfs_type, path_planning::map_height>,
    path_planning::map_width>, hash> test_maps;
  std::array<std::array<path_planning::kfs_type, path_planning::map_height>,
    path_planning::map_width> empty_map{};
  empty_map[1][5] = path_planning::kfs_type::falsekfs; // Set an obstacle at (1, 5)
  auto func = [&](int idx) {
    if (idx == -1) {
      for (int i = 0; i < path_planning::map_width; ++i) {
        for (int j = 2; j < path_planning::map_height - 1; ++j) {
          if (empty_map[i][j] != path_planning::kfs_type::empty) {
            continue; // Skip already set nodes
          }
          empty_map[i][j] = path_planning::kfs_type::falsekfs;
          setup_map(idx + 1);
          empty_map[i][j] = path_planning::kfs_type::empty;
        }
      }
    }
    if (idx >= types.size()) {
      if (test_maps.find(empty_map) == test_maps.end()) {
        test_maps.insert(empty_map);
      }
      return;
    }
    for (int i = 0; i < path_planning::map_width; ++i) {
      for (int j = 1; j < path_planning::map_height - 1; ++j) {
        if (empty_map[i][j] != path_planning::kfs_type::empty) {
          continue; // Skip already set nodes
        }
        if (i == 1 && (j >= 2 && j <= 3) && types[idx] == path_planning::kfs_type::r1kfs) {
          continue; // Skip setting r1kfs on the path
        }
        empty_map[i][j] = types[idx];
        setup_map(idx + 1);
        empty_map[i][j] = path_planning::kfs_type::empty;
      }
    }
  };
  setup_map = func;

  std::cout << "Generating test maps...\n";
  func(-1); // Start generating test maps with -1 to indicate the initial call

  std::cout << "Test maps generated: " << test_maps.size() << "\n";
  std::cout << "Running tests...\n";

  path_planning::point start{1, 1};
  path_planning::point end1{0, 5};
  path_planning::point end2{2, 5};

  for (const auto& map : test_maps) {
    for (int i = 0; i < path_planning::map_width; ++i) {
      for (int j = 0; j < path_planning::map_height; ++j) {
        planner.set_kfs_type({i, j}, map[i][j]);
      }
    }

    std::cout << "Test case " << ++test_count << ":\n";

    auto path1 = planner.find_path(start, end1);
    auto path2 = planner.find_path(start, end2);
    std::queue<path_planning::a_star_node> path;
    if (path1.empty() && path2.empty()) {
      std::cout << "No path found\n";
      continue;
    }
    else if (path1.empty()) {
      path = std::move(path2);
    } else if (path2.empty()) {
      path = std::move(path1);
    } else {
      if (path1.back().f_cost() <= path2.back().f_cost()) {
        path = std::move(path1);
      } else {
        path = std::move(path2);
      }
    }

    // Remove the obstacle for visualization
    planner.set_kfs_type({1, 5}, path_planning::kfs_type::empty);

    std::cout << "Grid:\n";
    for (int j = 0; j < path_planning::map_height; ++j) {
      for (int i = 0; i < path_planning::map_width; ++i) {
        std::cout << static_cast<int>(planner.get_kfs_type({i, j})) << " ";
      }
      std::cout << "\n";
    }

    std::cout << "Path found:\n";
    while (!path.empty()) {
      const auto &node = path.front();
      std::cout << "(" << node.p.x << ", " << node.p.y << ") - Type: "
                << static_cast<int>(node.type) << "\n";
      path.pop();
    }
  }
}
