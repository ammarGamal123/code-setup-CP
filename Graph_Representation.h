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
=> 2 ways to represent graph
1- adjcency matrix
2- adjcency list
*/
bool matrix[sz][sz];
void solve () {
    int nodes , edges;
    cin >> nodes >> edges;
    vector <vector <int> > adj_dynamic (nodes + 1);
    for (int i = 0; i < edges; i++){
        int x , y;
        cin >> x >> y;
        matrix[x][y] = 1;
        adj_dynamic[x].emplace_back(y);
    }
    cout << "Matrix " << endl;
    for (int i = 1; i <= nodes; i++){
        for (int j = 1; j <= nodes; j++){
            cout << matrix[i][j] << " " ;
        }
        cout << endl;
    }
    cout << "-----------------------" << endl;
    cout << "Adjcency list" << endl;
    for (int i = 1; i < adj_dynamic.size(); i++){
        cout << i << " { " ;
        for (int j = 0; j < adj_dynamic[i].size(); j++){
            cout << adj_dynamic[i][j] << " ";
        }
        cout << " } " << endl;
    }
}



