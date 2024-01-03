//
// Created by ammarhammad on 14/05/23.
//

#ifndef ATTACK_CP_REUCURSION_BACKTRACKING_H
#define ATTACK_CP_REUCURSION_BACKTRACKING_H

#endif //ATTACK_CP_REUCURSION_BACKTRACKING_H

#include <bits/stdc++.h>

using namespace std;
using int64 = int64_t;
#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbeg    in(), (x).rend()
#define int    long long
#define lll __int128
#define ordered_set tree<int, null_type,less<int>, rb_tree_tag,tree_order_statistics_node_update>
#define Ceil(n, m) (((n) / (m)) + ((n) % (m) ? 1 : 0))
#define endl '\n'
#define NeedForSpeed ios_base::sync_with_stdio(false) , cin.tie(nullptr), cout.tie(nullptr);
const int64 INF = 1'000'000'000LL + 100;
const long double PI = acos(-1);
const int N1 = 2e5 + 7, Mod = 1000000007 , inf = 1e9 , bitstr= 27;
const int NN = 1e7 + 5 , OO = 0x3F3F3F3F;


// time complexity = 2 ^ n
// a = {3 , 1 , 2}
void print_all_subsequence (vector <int> &ds ,vector <int> &a , int n , int idx){
    if (idx >= n){
        for (auto i : ds)
            cout << i << " ";
        if (ds.empty()){
            cout << "{}" << endl;
        }
        cout << endl;
        return;
    }
    // take or pick the particular index into the subsequence
    ds.push_back(a[idx]);
    print_all_subsequence(ds , a , n , idx + 1);
    ds.pop_back();
    // not pick, or not take condition, this element is not added to your subsequence
    print_all_subsequence(ds , a , n , idx + 1);
}


int32_t main (){

    return 0;
}