#include <bits/stdc++.h>
using namespace std;
const int sz = 1e3+10;
//int matrix [sz][sz];
const int OO = 0x3f3f3f3f;
int dis[sz];
vector<int>adj[sz];
bool matrix [sz][sz];
void bfs (int node ){
    memset(dis , OO , sizeof(dis));
    queue <int> q;
    q.push(node);
    dis[node] = 1;
    while (!q.empty()){
        int p = q.front();
        q.pop();
        for (int i = 0;i < adj[p].size();i ++){
            int child = adj[p][i];
            if (dis[child] == OO)
            {
                dis[child] = dis[p] + 1;
                q.push(child);
            }
        }
    }
}
void representation (){
    int n, m;
    cin >> n >> m;
    vector<vector<int>> adj_dynamic(n + 1);
    for (int i = 0; i < m; i++) {
        int x, y;
        cin >> x >> y;
        matrix[x][y] = 1;
        matrix[y][x] = 1;
        adj_dynamic[x].emplace_back(y);
        adj_dynamic[y].emplace_back(x);
    }
    cout << "Matrix " << endl;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
    cout << "-----------------------" << endl;
    cout << "Adjacency list" << endl;
    for (int i = 1; i < adj_dynamic.size(); i++) {
        cout << i << " = { ";
        for (int j = 0; j < adj_dynamic[i].size(); j++) {
            cout << adj_dynamic[i][j] << " ";
        }
        cout << "}" << endl;
    }
}
vector <int> graph [55];
bool vis[55];
int cnt = 1;

void dfs (int node){
    vis[node] = true;
    for (auto x : graph[node]){
        if (!vis[x])
            dfs(x) , cnt >>= 1;
    }
}
int main() {

    return 0;
}