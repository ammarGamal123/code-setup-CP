//
// Created by ammarhammad on 16/03/23.
//

#ifndef ATTACK_CP_BFS_H
#define ATTACK_CP_BFS_H

#endif //ATTACK_CP_BFS_H
#include <bits/stdc++.h>
using namespace std;
#define all(v)				((v).begin()), ((v).end())
#define sz(v)				((int)((v).size()))
#define clr(v, d)			memset(v, d, sizeof(v))
#define rep(i, v)		for(int i=0;i<sz(v);++i)
#define lp(i, n)		for(int i=0;i<(int)(n);++i)
#define lpi(i, j, n)	for(int i=(j);i<(int)(n);++i)
#define lpd(i, j, n)	for(int i=(j);i>=(int)(n);--i)

typedef long long         ll;
const int OO = (int)1e6;
const double EPS = (1e-7);
int dcmp(double x, double y) {	return fabs(x-y) <= EPS ? 0 : x < y ? -1 : 1;	}

#define pb					push_back
#define MP					make_pair
#define P(x)				cout<<#x<<" = { "<<x<<" }\n"
typedef long double   	  ld;
typedef vector<int>       vi;
typedef vector<double>    vd;
typedef vector< vi >      vvi;
typedef vector< vd >      vvd;
typedef vector<string>    vs;
const int sz = 1001;

/*
==============>Code of BFS<==============
*/

using namespace std;

vector<bool> v;
vector<vector<int> > g;

void edge(int a, int b)
{
    g[a].push_back(b);

    // for undirected graph add this line
    // g[b].pb(a);
}

void bfs(int u) {
    queue<int> q;

    q.push(u);
    v[u] = true;

    while (!q.empty()) {

        int f = q.front();
        q.pop();

        cout << f << " ";

        // Enqueue all adjacent of f and mark them visited
        for (auto i = g[f].begin(); i != g[f].end(); i++) {
            if (!v[*i]) {
                q.push(*i);
                v[*i] = true;
            }
        }
    }
}


vector <int> BFS(int start , vector <vector <int>> &adjList) {
    vector<int> len(adjList.size(), INT_MAX);
    queue<pair<int, int>> q;
    q.push(make_pair(start, 0));
    len[start] = 1;
    int cur, dep;
    while (!q.empty()) {
        pair<int, int> p = q.front();
        q.pop();
        cur = p.first, dep = p.second;
        for (int i = 0; i < adjList[cur].size(); i++) {
            if (len[adjList[cur][dep]] == INT_MAX) {
                q.push(make_pair(adjList[cur][i], dep + 1));
                len[adjList[cur][i]] = dep + 1;
            }
        }
    }
    return len; // cur is the furthest node from s with depth dep
}
vector<int> BFS2(int s, vector<vector<int> > & adjList) {
    vector<int> len(sz(adjList), OO);
    queue<int> q;
    q.push(s), len[s] = 0;

    int dep = 0, cur = s, sz = 1;
    for (; !q.empty(); ++dep, sz = q.size()) {
        while (sz--) {
            cur = q.front(), q.pop();
            rep(i, adjList[cur]) if (len[adjList[cur][i]] == OO)
                    q.push(adjList[cur][i]), len[adjList[cur][i]] = dep + 1;
        }
    }
    return len;    //cur is the furthest node from s with depth dep
}

vector<int> BFSPath(int s, int d, vector<vector<int> > & adjList) {
    vector<int> len(sz(adjList), OO);
    vector<int> par(sz(adjList), -1);
    queue<int> q;
    q.push(s), len[s] = 0;

    int dep = 0, cur = s, sz = 1;
    bool ok = true;

    for (; ok && !q.empty(); ++dep, sz = q.size()) {
        while (ok && sz--) {
            cur = q.front(), q.pop();
            rep(i, adjList[cur]) if (len[adjList[cur][i]] == OO) {
                    q.push(adjList[cur][i]), len[adjList[cur][i]] = dep + 1, par[adjList[cur][i]] = cur;

                    if (adjList[cur][i] == d)    // we found target no need to continue
                    {
                        ok = false;
                        break;
                    }
                }
        }
    }

    vector<int> path;
    while (d != -1) {
        path.push_back(d);
        d = par[d];
    }

    reverse(all(path));

    return path;
}


// All is done by 1 BFS
// 1-1, 1-m
// m-1 -> Reverse it, start with target and go to sources
// m-m -> Set all start nodes in Q, and find targets

// Testing a graph for bipartiteness

// Edge Split, Vertex Split Tricks

// N=5
// 3	3
// 3 - 6, 6 - 7, 7 - 8
// in(3) = 3, out(3) = 8
// sp(1, 3) = sp( in(1), out(3)) = (1, 8)

int main() {
#ifndef ONLINE_JUDGE
    freopen("c.in", "rt", stdin);
    //freopen(".txt", "wt", stdout);
#endif

    int n, e;

    cin >> n >> e;

    vector<vector<int> > adj(n);

    lp(i, e) {
        int from, to;
        cin >> from >> to;
        adj[from - 1].push_back(to - 1);
    }

    vector<int> p = BFSPath(0, 4, adj);
    rep(i, p)cout << p[i] + 1 << " ";

    return 0;
}
