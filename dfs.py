from collections import defaultdict

class Graph:
    def __init__(self):
        self.adjacency_list = defaultdict(list)
        self.temp = None

    def add_edge(self, u, v):
        self.adjacency_list[u].append(v)

    def dfs_util(self, vertex, visited):
        visited.add(vertex)
        self.temp.append(vertex)
        for neighbor in self.adjacency_list[vertex]:
            if neighbor not in visited:
                self.dfs_util(neighbor, visited)

    def dfs(self, start_vertex):
        visited_set = set()
        self.temp = []
        self.dfs_util(start_vertex, visited_set)


if __name__ == "__main__":
    pass