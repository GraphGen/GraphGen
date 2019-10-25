mkdir -p bin

g++ -std=c++11 dfs_code.cpp -o bin/dfs_code -O3
g++ -std=c++11 metrics/orca/orca.cpp -o metrics/orca/orca -O3
g++ -std=c++11 isomorph.cpp -O3 -o bin/sub_iso -fopenmp -lboost_graph
g++ -std=c++11 unique.cpp -O3 -o bin/unique -fopenmp -lboost_graph