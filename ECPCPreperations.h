//
// Created by ammarhammad on 11/08/23.
//

#ifndef ATTACK_CP_ECPCPREPERATIONS_H
#define ATTACK_CP_ECPCPREPERATIONS_H

#endif //ATTACK_CP_ECPCPREPERATIONS_H
#include <bits/stdc++.h>

using namespace std;
#define int long long
#define ll long long
using namespace std;
using int64 = int64_t;
#define all(x) (x).begin(), (x).end()
#define ll long long
#define rall(x) (x).rbegin(), (x).rend()
#define output_vector(a) for (auto i : a) cout << i << " ";
#define int    long long
#define no return void (cout << "NO" << endl)
#define yes return void (cout << "YES" << endl)
#define lll __int128
#define ordered_set tree<int, null_type,less<int>, rb_tree_tag,tree_order_statistics_node_update>
#define Ceil(n, m) (((n) / (m)) + ((n) % (m) ? 1 : 0))
#define getunique(v) {sort(v.begin(), v.end()); v.erase(unique(v.begin(), v.end()), v.end());}
#define endl '\n'
#define NeedForSpeed ios_base::sync_with_stdio(false) , cin.tie(nullptr), cout.tie(nullptr);
const int64 INF = 1000000000LL + 100;
const long double PI = acos(-1);
const int N1 = 2e5 + 7, Mod = 1000000007 , inf = 1e9 , bitstr = 27;
const int NN1 = 1000 + 5 , OO = 0x3F3F3F3F;
const int MAXN = 2e5 + 5;

// Sparse Table

const int N=1e5+5;
ll s[N],dp[N][22];int n,q,LOG[N];

// you can manipulate here with the merge function you can make it with min (a , b) , Xor (a , b),
// gcd (a , b) , lcm (a , b);
ll merge(ll a,ll b)
{
    return max(a,b);
}
void build() {
    for (int i = 0; i < n; i++) {
        dp[i][0] = s[i];
    }
    for (int mask = 1; (1 << mask) <= n; mask++) {
        for (int i = 0; i + (1 << mask) <= n; i++) {
            dp[i][mask] = merge(dp[i][mask - 1], dp[i + (1 << (mask - 1))][mask - 1]);
        }
    }
}
ll query(int l,int r) {
    int mask = LOG[r - l + 1];
    return merge(dp[l][mask], dp[r - (1 << mask) + 1][mask]);
}
void preCount() {
    LOG[1] = 0;
    for (int i = 2; i < N; i++) {
        LOG[i] = LOG[i >> 1] + 1;
    }
}
/*
signed main() {
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    preCount(); here is important thing that preCount(); function should be called out of testcases
    cin >> n >> q;
    for (int i = 0; i < n; i++) {
        cin >> s[i];
    }
    build();
    while (q--) {
        int l, r;
        cin >> l >> r;
        cout << query2(--l, --r) << '\n';
    }
    return 0;
}
 */

// Fenwick Tree is much faster than sparse table
struct FenwickTree {
    ll bit[N] = {};
    /* here in add function the val parameter should be the value not the needed element
     * for ex if the a[x] = y , and we update a[x] = z , the parameter should be called like that
     * add (idx , z - y);
     * not add (idx , z);
     * */

    void add(int idx, ll val) {
        while (idx < N) {
            bit[idx] += val;
            idx += idx & -idx;
        }
    }

    ll query(int idx) {
        ll ret = 0;
        while (idx > 0) {
            ret += bit[idx];
            idx -= idx & -idx;
        }
        return ret;
    }

    ll prefix(int l, int r) {
        return query(r) - query(l - 1);
    }
};


// segment tree should be called with n + 5 size like that
// SegmentTree st (n + 5);

struct SegmentTree {
private:
    vector<int> seg;

    // the skip value should be the initial value for the merge function below
    /* for ex : if the merge calc the max between two numbers the initial value for the skip
     * should be INT_MIN and so on for every merge type
     * */

    int sz, skip = INT_MAX;

    int merge(int a, int b) {
        return min(a, b);
    }

    void update(int l, int r, int node, int idx, int val) {
        if (l == r) {
            seg[node] = val;
            return;
        }
        int mid = l + r >> 1;
        if (mid < idx) {
            update(mid + 1, r, 2 * node + 2, idx, val);
        } else {
            update(l, mid, 2 * node + 1, idx, val);
        }
        seg[node] = merge(seg[2 * node + 1], seg[2 * node + 2]);
    }

    int query(int l, int r, int node, int lx, int rx) {
        if (l > rx || r < lx)return skip;
        if (l >= lx && r <= rx)return seg[node];
        int mid = l + r >> 1;
        int a = query(l, mid, 2 * node + 1, lx, rx);
        int b = query(mid + 1, r, 2 * node + 2, lx, rx);
        return merge(a, b);
    }

public:
    SegmentTree(int n) {
        sz = 1;
        while (sz <= n)sz <<= 1;
        seg = vector<int>(sz << 1, skip);
    }

    void update(int idx, int val) {
        update(0, sz - 1, 0, idx, val);
    }

    int query(int l, int r) {
        return query(0, sz - 1, 0, l, r);
    }
};

// Dijkstra for graph to get the shortest path between nodes
// note that it sould contain a start point

// const int N=1e5+5;
vector<pair<int,int>>adj[N];vector<int>cost(N,-1);
void dijkstra(int start) {
    priority_queue<pair<int, int>, deque<pair<int, int>>, greater<pair<int, int>>> pq;

    pq.push({0, start});

    while (pq.size()) {
        pair<int, int> p = pq.top();
        pq.pop();
        int node = p.second, nodecost = p.first;

        if (cost[node] != -1) {
            continue;
        }

        cost[node] = nodecost;

        for (auto [node2, cost2]: adj[node]) {
            if (cost[node2] == -1) {
                pq.push({nodecost + cost2, node2});
            }
        }
    }
}
///  graph

const int sz = 1e3+10;
//int matrix [sz][sz];
const int OO1 = 0x3f3f3f3f;
int dis[sz];
vector<int>adj1[sz];
bool matrix1 [sz][sz];
void bfs (int node ){
    memset(dis , OO1 , sizeof(dis));
    queue <int> q;
    q.push(node);
    dis[node] = 1;
    while (!q.empty()){
        int p = q.front();
        q.pop();
        for (int i = 0;i < adj1[p].size();i ++){
            int child = adj1[p][i];
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
        matrix1[x][y] = 1;
        matrix1[y][x] = 1;
        adj_dynamic[x].emplace_back(y);
        adj_dynamic[y].emplace_back(x);
    }
    cout << "Matrix " << endl;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            cout << matrix1[i][j] << " ";
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

///  End of graph

// Ordered set , multiset
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
#define ordered_set tree<int, null_type,less<>, rb_tree_tag,tree_order_statistics_node_update>// set

typedef tree<int, null_type,less_equal<int>, rb_tree_tag,tree_order_statistics_node_update> ordered_multiset;
#define ll long long
void myErase(ordered_set &t, int v){
    int rank = t.order_of_key(v);//Number of elements that are less than v in t
    ordered_set::iterator it = t.find_by_order(rank); //Iterator that points to the (rank+1)th element in t
    t.erase(it);
}


/*
 * int main()
{
    ios_base::sync_with_stdio(0);cin.tie(0);cout.tie(0);
    ordered_multiset st;
    int n;
    cin >> n;
    long long sum = 0;
    for (int i = 0 ; i < n;i++)
    {
        int x;
        cin >> x;
        st.insert(x);
        // to get the number of elements greater than me
        sum += st.order_of_key(x);
    }
    cout << sum << endl;
    return 0;
}
 */


/* to get the number of divisors if n <= 1e18 it just get how many divisors for the number
 * not the actual value of each divisors
 */
// ==> sqrt (1e18) -> (1e9)
// ==> thisCode < (1e8)


// to know the first and last occurance of each character but there is a thing
// the string should be sorted

void store (){
    vector <string> vs(n);
    vector<vector<pair<int, int>>>freq(n, vector<pair<int, int>>(26));
    for (int i = 0;i < n;i++){
        int j = 0;
        while (j < vs[i].size()){
            int u = j;
            j ++;
            while (j < vs[i].size() && vs[i][j] == vs[i][j - 1]){
                j ++;
            }
            freq[i][vs[i][u] - 'a'].first = u + 1;
            freq[i][vs[i][u] - 'a'].second = j;
        }
    }
}

//   here to compress vector
//   for ex : [ 1 , 1 , 1 , 2 , 3 , 3 , 2 , 5]
// it will return this array [ 0 , 0 , 0 , 1 , 2 , 2 , 1 , 3]
// and it is fast

void compress(vector<ll>&a,int start) {
    int n = a.size();
    vector<pair<ll, ll>> pairs(n);
    for (int i = 0; i < n; i++) {
        pairs[i] = {a[i], i};
    }
    sort(pairs.begin(), pairs.end());
    int nxt = start;
    for (int i = 0; i < n; i++) {
        if (i > 0 && pairs[i - 1].first != pairs[i].first) {
            nxt++;
        }
        a[pairs[i].second] = nxt;
    }
}

///// Trie to like countPrefix of each string and bring the result

struct Trie {
    struct Node {
        Node *child[26];
        int IsEnd, Prefix;

        Node() {
            memset(child, 0, sizeof child);
            IsEnd = Prefix = 0;
        }
    };

    Node *root = new Node();
    // here U can delete the reference and add it
    void insert(string s) {
        Node *cur = root;
        for (auto it: s) {
            int idx = it - 'a';
            if (cur->child[idx] == 0) {
                cur->child[idx] = new Node();
            }
            cur = cur->child[idx];
            cur->Prefix++;
        }
        cur->IsEnd++;
    }

    bool SearchWord(string &s) {
        Node *cur = root;
        for (auto it: s) {
            int idx = it - 'a';
            if (cur->child[idx] == 0)return 0;
            cur = cur->child[idx];
        }
        return cur->IsEnd;
    }

    int CountWord(string &s) {
        Node *cur = root;
        for (auto it: s) {
            int idx = it - 'a';
            if (cur->child[idx] == 0)return 0;
            cur = cur->child[idx];
        }
        return cur->IsEnd;
    }

    int CountPrefix(string &s) {
        Node *cur = root;
        for (auto it: s) {
            int idx = it - 'a';
            if (cur->child[idx] == 0)return 0;
            cur = cur->child[idx];
        }
        return cur->Prefix;
    }
};
void solve1 () {
    Trie t;
    t.insert ("abx");
    t.insert ("abc");
    t.insert ("lxa");
    t.insert ("cab");
    string target = "c";
    cout << t.CountPrefix(target) << endl;
}
/// End of Trie

///// The Beginning of Hashing

struct Hashing {
private:
    int mod1 = 1e9 + 7, mod2 = 2e9 + 11;
    ll base1, base2, h1, h2, inv1, inv2, *pw1, *pw2, len;
    deque<char> d;

    ll power(ll a, ll b, ll m) {
        ll ans = 1;
        while (b > 0) {
            if (b & 1) {
                ans = (ans * a) % m;
            }
            a = (a * a) % m;
            b >>= 1;
        }
        return ans;
    }

public:
    Hashing(int sz, ll x = 31, ll y = 37) {
        // if you gonna call Hashing too much you should remove this segment and put it away of this struct
        base1 = x;
        base2 = y;
        h1 = h2 = len = 0;
        inv1 = power(x, mod1 - 2, mod1);
        inv2 = power(y, mod2 - 2, mod2);
        pw1 = new ll[sz + 1];
        pw2 = new ll[sz + 1];
        pw1[0] = pw2[0] = 1;
        for (int i = 1; i <= sz; i++) {
            pw1[i] = (x * pw1[i - 1]) % mod1;
            pw2[i] = (y * pw2[i - 1]) % mod2;
        }
    }

    void push_back(char x) {
        x = x - 'a' + 1;
        h1 = (h1 * base1) % mod1;
        h1 = (h1 + x) % mod1;
        h2 = (h2 * base2) % mod2;
        h2 = (h2 + x) % mod2;
        len++;
        d.emplace_back(x);
    }

    void push_front(char x) {
        x = x - 'a' + 1;
        h1 = (h1 + (x * pw1[len]) % mod1) % mod1;
        h2 = (h2 + (x * pw2[len]) % mod2) % mod2;
        len++;
        d.emplace_front(x);
    }

    void pop_back() {
        if (len == 0)return;
        char x = d.back();
        d.pop_back();
        h1 = (h1 - x + mod1) % mod1;
        h1 = (h1 * inv1) % mod1;
        h2 = (h2 - x + mod2) % mod2;
        h2 = (h2 * inv2) % mod2;
        len--;
    }

    void pop_front() {
        if (len == 0)return;
        char x = d.front();
        d.pop_front();
        len--;
        h1 = ((h1 - x * pw1[len] % mod1) + mod1) % mod1;
        h2 = ((h2 - x * pw2[len] % mod2) + mod2) % mod2;
    }

    void clear() {
        h1 = h2 = len = 0;
        d.clear();
    }

    bool operator==(const Hashing &H) const {
        return H.h1 == h1 && H.h2 == h2;
    }

    string GetString() {
        return string(d.begin(), d.end());
    }

    pair<int, int> GetHash() {
        return {h1, h2};
    }
};

//// The End of Hashing

/// The Beginning of KMP

struct KMP {
    int longestPrefix[N] = {};
    vector<int> ans;

    void calcPrefix(string patern) {
        int n = patern.size();
        for (int i = 1, idx = 0; i < n; i++) {
            while (idx > 0 && patern[idx] != patern[i])  idx = longestPrefix[idx - 1];
            if (patern[i] == patern[idx])idx++;
            longestPrefix[i] = idx;
        }
    }

    void kmp(string s, string pat) {
        int n = s.size(), m = pat.size();
        calcPrefix(pat);
        for (int i = 0, idx = 0; i < n; i++) {
            while (idx > 0 && s[i] != pat[idx])idx = longestPrefix[idx - 1];
            if (s[i] == pat[idx])idx++;
            if (idx == m)ans.push_back(i - m + 1), idx = longestPrefix[idx - 1];
        }
    }
};
void solve2 () {
    KMP k;
    k.kmp("abaaababaaab" , "aba");
    // to the index for each repatation of the pattern
    for (auto i : k.ans) cout << i << " ";
}
/// The End of KMP

// The Beginning of SegmentHashing

const int N3 = 1e5+5 , mod1 = 1e9+7 , mod2 = 2e9+11;
ll base1=31,base2=37,pw1[N+1],pw2[N+1],inv1[N+1],inv2[N+1];
ll powmod(ll a,ll b,ll m)
{
    ll ans=1;
    while(b>0)
    {
        if(b&1)
        {
            ans=(ans*a)%m;
        }
        a=(a*a)%m;
        b>>=1;
    }
    return ans;
}
void init()
{
    pw1[0]=pw2[0]=inv1[0]=inv2[0]=1;
    int temp1=powmod(base1,mod1-2,mod1);
    int temp2=powmod(base2,mod2-2,mod2);
    for(int i=1;i<N;i++)
    {
        pw1[i]=(base1*pw1[i-1])%mod1;
        pw2[i]=(base2*pw2[i-1])%mod2;
        inv1[i]=(inv1[i-1]*temp1)%mod1;
        inv2[i]=(inv2[i-1]*temp2)%mod2;
    }
}
struct HashingSegmentTree {
private:
    vector<pair<int, int>> seg;
    int sz;

    pair<int, int> merge(pair<int, int> l, pair<int, int> r) {
        pair<int, int> ret = l;
        ret.first = (ret.first + r.first) % mod1;
        ret.second = (ret.second + r.second) % mod2;
        return ret;
    }

    void update(int l, int r, int node, int idx, int ch) {
        if (l == r) {
            seg[node] = {(ch * pw1[idx]) % mod1, (ch * pw2[idx]) % mod2};
            return;
        }
        int mid = l + r >> 1;
        if (idx <= mid)update(l, mid, 2 * node + 1, idx, ch);
        else update(mid + 1, r, 2 * node + 2, idx, ch);
        seg[node] = merge(seg[2 * node + 1], seg[2 * node + 2]);
    }

    pair<int, int> query(int l, int r, int node, int lx, int rx) {
        if (l >= lx && r <= rx) {
            return seg[node];
        }
        if (l > rx || r < lx)return {0, 0};
        int mid = l + r >> 1;
        pair<int, int> lft = query(l, mid, 2 * node + 1, lx, rx);
        pair<int, int> rgt = query(mid + 1, r, 2 * node + 2, lx, rx);
        return merge(lft, rgt);
    }

public:
    HashingSegmentTree(int n) {
        sz = 1;
        while (sz <= n)sz *= 2;
        seg = vector<pair<int, int>>(sz * 2);
    }

    void update(int idx, char ch) {
        update(0, sz - 1, 0, idx, ch - 'a' + 1);
    }

    pair<int, int> query(int l, int r) {
        pair<int, int> ret = query(0, sz - 1, 0, l, r);
        ret.first = (ret.first * inv1[l - 1]) % mod1;
        ret.second = (ret.second * inv2[l - 1]) % mod2;
        return ret;
    }
};
bool isPalindrome(HashingSegmentTree &a,HashingSegmentTree &b,int &l,int &r,int &n) {
    return (a.query(l, r) == b.query(n - r + 1, n - l + 1));
}

/// The End Of SegmentHashing

/////// The Begin Bitmasking

void print_Number(int n){
    if (n <= 1) {
        cout << n;
        return;
    }
    print_Number(n >> 1);
    cout << (n & 1) ;
}
int countBits_1(int n ){
    int cnt = 0;
    while (n){
        cnt += (n & 1);
        n >>= 1;
    }
    return cnt ;
}
bool isPowerOfFour(int n) {
    return !(n & (n - 1)) && (n & 0x55555555);
    //check the 1-bit location;
}

bool checkBit_index(int num  , int index){
    return (num >> index) & 1;
}
int setBit_1(int num , int index){
    return num | ( 1 << index);
}
int setBit_0(int num , int index){
    return num & ~ (1 << index);
}
int flibBit(int num , int index){
    return num ^ (1 << index);
}
int check_Num_Pow_Of_2_Or_Not(int num){
    return !(num & num -1);
}
int largest_bit(int x) {
    return x == 0 ? -1 : 31 - __builtin_clz(x);
}
int count_1 (int num ){
    int cnt {};
    while (num){
        cnt ++;
        num &= (num - 1);
    }
    return cnt;
}
int find_leftmost_bit (int Xor){
    int cnt {};
    while (Xor){
        if (Xor & 1) break;
        cnt ++;
        Xor >>= 1;
    }
    return cnt;
}
int lastBitValue(int n){
    return n & ~(n-1);
}
void print_all_subsets (vector <int> &a){
    int n = a.size();
    for (int i = 0;i <= (1 << n) - 1; i++){
        for (int j = 0; j < n; j++){
            // check bit is 1 to print it
            if ((1 << j) & i)
                cout << a[j] << " ";
        }
        cout << endl;
    }
}

// how to make data structure which add , remove , print element in O(1)
struct great_ds {
    int mask = 0;
    void add (int x){
        mask |= (1 << (x));
    }
    void remove (int x){
        mask &= (~(1 << x));
    }
    void print (){
        for (int bit = 0; bit <= 60; bit++){
            if ( (mask >> bit) & 1){
                cout << bit << " ";
            }
        }
        cout << endl;
    }
};

int binaryToDecimal(string n)
{
    string num = n;
    int dec_value = 0;

    // Initializing base value to 1, i.e 2^0
    int base = 1;

    int len = num.length();
    for (int i = len - 1; i >= 0; i--) {
        if (num[i] == '1')
            dec_value += base;
        base = base * 2;
    }

    return dec_value;
}

//// The End of Bitmasking



////// The begin of Number theory






int sum_odd (int n ){
    int ans = (n + 1) / 2;
    return ans * ans;
}
// to convert binary string to decimal integer
int binaryToDecimal1(string n)
{
    string num = n;
    int dec_value = 0;

    // Initializing base value to 1, i.e 2^0
    int base = 1;

    int len = num.length();
    for (int i = len - 1; i >= 0; i--) {
        if (num[i] == '1')
            dec_value += base;
        base = base * 2;
    }

    return dec_value;
}

long long fast_power(long long base, long long power) {
    long long result = 1;
    while(power > 0) {

        if(power & 1 ) { // Can also use (power & 1) to make code even faster
            result = (result*base) % Mod;
        }
        base = (base * base) % Mod;
        power = power / 2; // Can also use power >>= 1; to make code even faster
    }
    return result;
}
// The condition to make a triangle
/*/*
All you have to do is use the Triangle Inequality Theorem,
which states that the sum of two side lengths of a triangle
is always greater than the third side. If this is true for
all three combinations of added side lengths, then you will
have a triangle.
*/

// very fast code to get prime factorization
vector<long long> trial_division3(long long n) {
    vector<long long> factorization;
    for (int d : {2, 3, 5}) {
        while (n % d == 0) {
            factorization.push_back(d);
            n /= d;
        }
    }
    static array<int, 8> increments = {4, 2, 4, 2, 4, 6, 2, 6};
    int i = 0;
    for (long long d = 7; d * d <= n; d += increments[i++]) {
        while (n % d == 0) {
            factorization.push_back(d);
            n /= d;
        }
        if (i == 8)
            i = 0;
    }
    if (n > 1)
        factorization.push_back(n);
    return factorization;
}

vector<int> primeFactor (int n){
    vector<int> ret;
    for (int i = 2;i <= n / i;i+= 1 + (i & 1)){
        while (n % i == 0){
            ret.emplace_back(i);
            n /= i;
        }
    }
    if (n > 1) ret.emplace_back(n);
    return ret;
}

vector < pair<int,int> > prime_count (int n){
    vector<pair< int , int >> all;
    for (int p = 2; p * p <= n; p++) {
        if (n % p == 0) {
            int count = 0;
            while (n % p == 0) {
                count++;
                n /= p;
            }
            all.emplace_back(p, count);
        }
    }
    if (n > 1) all.emplace_back(n , 1);
    return all;
}

// check prime or not

void How_to_remove_duplicates (){
    vector <int> a {1 , 1 , 3 , 1 , 2 , 2 , 5 , 5};
    auto last = std::unique(a.begin(), a.end());

    // Finally, erase the duplicate elements from the vector
    a.erase(last, a.end());
}
int get_divisors(int n) {

    vector <int> divisors;

    for (int i = 1; i * i <= n; ++i) {
        if (n % i == 0) {
            divisors.emplace_back(i);
            if (i != n / i) {
                divisors.emplace_back(n / i);
            }
        }
    }
    return divisors.size();
}
int add(int a, int b)
{
    long long x = a + b;
    if (x >= Mod) x -= Mod;
}
int sub (int a , int b){
    long long x = a - b;
    if (x < 0) x += Mod;
}
int mul(int a, int b)
{
    return ((int)a * b) % Mod;
}
//  divide a by b, modulo-style
int div_a_by_b (int a , int b) {
    long long b_inverse = fast_power(b, Mod - 2);
    long long x = (a * b_inverse) % Mod;
}
//Computing all factorials up to n in O(n)
void factorial (int n ){
    long long fact[n + 1];
    fact[0] = fact[1] = 1;
    for (int i = 2; i <= n; i++) {
        fact[i] = (fact[i - 1] * i) % Mod;
    }
}
//Computing all inverse factorials up to $n$ in $O(n)$
int inverse_factorials (int n ) {
    long long fact[n + 1];
    fact[0] = fact[1] = 1;
    for (int i = 2; i <= n; i++) {
        fact[i] = (fact[i - 1] * i) % Mod;
    }

    long long ifact[n + 1];
    ifact[n] = fast_power(fact[n], Mod - 2);
    for (int i = n - 1; i >= 0; i--) {
        ifact[i] = ((i + 1) * ifact[i + 1]) % Mod;
    }
}
int count_bits(int n)
{
    int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}


double degree_raduis (){
    double degree = 45.0;
    double radius = degree * PI / (180);
    double get_sin = sin(radius);
}
int sumNatural(int n)
{
    int sum = (n * (n + 1));
    return sum;
}

// Function to return sum
// of even numbers in range L and R
int sumEven(int l, int r)
{
    return sumNatural(r/2) - sumNatural((l-1) / 2);
}
int fast_power1(int n , int p ){
    int result = 1;
    while (p > 0){ if (p % 2){result = (result  *  n ) ;}
        n = ((n ) * (n ));  p >>= 1;}
    return result;
}
int floorSqrt(int x)
{
    // Base cases
    if (x == 0 || x == 1)
        return x;

    // Starting from 1, try all numbers until
    // i * i is greater than or equal to x.
    int i = 1, result = 1;
    while (result <= x) {
        i++;
        result = i * i;
    }
    return i - 1;
}
vector<int> Prime_factorization(int n){
    vector<int> Prime;
    for (int i = 2; i * i <= n; i++){
        while (n % i == 0) {
            n /= i;
            Prime. emplace_back(i);
        }}
    if (n != 1){Prime.push_back(n);} return Prime;
}
int gcd(int a ,int b){
    while(b != 0){int curr = a ; a = b; b = curr % b;}
    return a;
}
int lcm (int a , int b){
    return (a / gcd(a,b)) * b;
}
bool is_Prime(int n)
{
    // Corner cases
    if (n <= 1)
        return false;
    if (n <= 3)
        return true;

    if (n % 2 == 0 || n % 3 == 0)
        return false;

    for (int i = 5; i * i <= n; i = i + 6)
        if (n % i == 0 || n % (i + 2) == 0)
            return false;

    return true;
}
bool Euclidean_Distance(int a , int b , int a1 , int b1,int limit){
    return (limit * limit) >= ((a1 - a) * (a1 - a) + (b1 - b) * (b1 - b));
}
vector<int> divisors(int n ){
    vector<int>ret;
    for (int i = 1; i <= n  / i; i++) {
        if (n % i == false) {
            ret.emplace_back(i); if (i != n / i) ret.emplace_back(n / i);}}
    return ret;
}
const int NN = 1e6 + 7 ;
bool prime [NN];
bool sieve() {
    memset(prime, true, sizeof prime);
    prime[0] = prime[1] = false;
    for (int i = 2; i < NN / 2; i++)
        if (prime[i])
            for (int j = i * i; j < NN; j += i)
                prime[j] = false;
}
int comp[NN];
void modified_sieve(){
    iota (comp , comp+NN, 0);
    comp[0] = comp[1] = -1;
    for (int i = 2; i < NN / i; i++)
        if (comp[i] == i)
            for (int j = i * i; j < NN; j+=i)
                if (comp[j] == j)
                    comp[j] = i;
}
vector<int>factorize_log(int n ){
    vector<int>ret;
    while (n > 1 ){
        ret.emplace_back(comp[n]);
        n /= comp[n];
    }
    return ret;
}
vector<pair<int,int>>return_factorize_and_count(int n){
    vector<pair<int,int>>ret;
    while (n > 1){
        int cur = comp[n];
        int cnt = 0;
        while (n % cur == false){
            cnt ++;
            n /= cur;
        }
        ret.push_back({cur,cnt});
    }
    return ret;
}
// Euler's totient function
int phi(int n) {
    int result = n;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            while (n % i == 0)
                n /= i;
            result -= result / i;
        }
    }
    if (n > 1)
        result -= result / n;
    return result;
}

// Euler totient function from 1 to n in O (n loglog n)

void phi_1_to_n(int n) {
    vector<int> phi(n + 1);
    for (int i = 0; i <= n; i++)
        phi[i] = i;

    for (int i = 2; i <= n; i++) {
        if (phi[i] == i) {
            for (int j = i; j <= n; j += i)
                phi[j] -= phi[j] / i;
        }
    }
}
// Divisor sum property
// Finding the totient from 1 to n
// but the complexity is worse (n log n)
void phi_1_to_n2(int n) {
    vector<int> phi(n + 1);
    phi[0] = 0;
    phi[1] = 1;
    for (int i = 2; i <= n; i++)
        phi[i] = i - 1;

    for (int i = 2; i <= n; i++)
        for (int j = 2 * i; j <= n; j += i)
            phi[j] -= phi[i];
}
struct Factorizer {
    // Factorizer factorizer(prec_n, sp_bound, rng_seed);
    //    prec_n is the bound for sieve (inclusive)
    //    all numbers will first be checked on primes <= sp_bound (if prec_n >= sp_bound)
    //    factorization for one number ~1e18 takes ~13ms

    vector<int> min_prime;
    vector<int> primes;
    int prec_n;
    int sp_bound;

    Factorizer(int prec_n = 100, int sp_bound = 100, int64_t rng_seed = -1) :
            prec_n(max(prec_n, 3LL)),
            sp_bound(sp_bound),
            rng(rng_seed != -1 ? rng_seed : chrono::steady_clock::now().time_since_epoch().count()) {
        min_prime.assign(prec_n + 1, -1);
        for (int i = 2; i <= prec_n; ++i) {
            if (min_prime[i] == -1) {
                min_prime[i] = i;
                primes.push_back(i);
            }
            int k = min_prime[i];
            for (int j : primes) {
                if (j * i > prec_n) break;
                min_prime[i * j] = j;
                if (j == k) break;
            }
        }
    }

    bool is_prime(int64_t n, bool check_small = true) {
        if (n <= prec_n)
            return min_prime[n] == n;

        if (check_small) {
            for (int p : primes) {
                if (p > sp_bound || (int64_t)p * p > n) break;
                if (n % p == 0) return false;
            }
        }

        int s = 0;
        int64_t d = n - 1;
        while (d % 2 == 0) {
            ++s;
            d >>= 1;
        }
        for (int64_t a : {2, 325, 9375, 28178, 450775, 9780504, 1795265022}) {
            if (a >= n) break;
            int64_t x = mpow_long(a, d, n);
            if (x == 1 || x == n - 1)
                continue;
            bool composite = true;
            for (int i = 0; i < s - 1; ++i) {
                x = mul_mod(x, x, n);
                if (x == 1)
                    return false;
                if (x == n - 1) {
                    composite = false;
                    break;
                }
            }
            if (composite)
                return false;
        }
        return true;
    }

    vector<pair<int64_t, int>> factorize(int64_t n, bool check_small = true) {
        vector<pair<int64_t, int>> res;
        if (check_small) {
            for (int p : primes) {
                if (p > sp_bound) break;
                if ((int64_t)p * p > n) break;
                if (n % p == 0) {
                    res.emplace_back(p, 0);
                    while (n % p == 0) {
                        n /= p;
                        res.back().second++;
                    }
                }
            }
        }

        if (n == 1) return res;
        if (is_prime(n, false)) {
            res.emplace_back(n, 1);
            return res;
        }

        if (n <= prec_n) {
            while (n != 1) {
                int d = min_prime[n];
                if (res.empty() || res.back().first != d)
                    res.emplace_back(d, 0);
                res.back().second++;
                n /= d;
            }
            return res;
        }

        int64_t d = get_divisor(n);
        auto a = factorize(d, false);
        for (auto &[div, cnt] : a) {
            cnt = 0;
            while (n % div == 0) {
                n /= div;
                ++cnt;
            }
        }
        auto b = factorize(n, false);

        int ia = 0, ib = 0;
        while (ia < a.size() || ib < b.size()) {
            bool choosea;
            if (ia == a.size()) choosea = false;
            else if (ib == b.size()) choosea = true;
            else if (a[ia].first <= b[ib].first) choosea = true;
            else choosea = false;

            res.push_back(choosea ? a[ia++] : b[ib++]);
        }
        return res;
    }

private:
    mt19937_64 rng;
    int64_t rnd(int64_t l, int64_t r) {
        return uniform_int_distribution<int64_t>(l, r)(rng);
    }

    int64_t mpow_long(int64_t a, int64_t p, int64_t mod) {
        int64_t res = 1;
        while (p) {
            if (p & 1) res = mul_mod(res, a, mod);
            p >>= 1;
            a = mul_mod(a, a, mod);
        }
        return res;
    }

    int64_t mul_mod(int64_t a, int64_t b, int64_t mod) {
        int64_t res = a * b - mod * (int64_t)((long double)1 / mod * a * b);
        if (res < 0) res += mod;
        if (res >= mod) res -= mod;
        return res;
    }

    int64_t get_divisor(int64_t n) {
        auto f = [&](int64_t x) -> int64_t {
            int64_t res = mul_mod(x, x, n) + 1;
            if (res == n) res = 0;
            return res;
        };

        while (true) {
            int64_t x = rnd(1, n - 1);
            int64_t y = f(x);
            while (x != y) {
                int64_t d = gcd(n, abs(x - y));
                if (d == 0)
                    break;
                else if (d != 1)
                    return d;
                x = f(x);
                y = f(f(y));
            }
        }
    }
};
ll NumberOfDivisors(ll n) {
    int primes[] = {2, 3, 5, 7, 11, 13, 17, 19};
    ll num = 1, ans = 1;
    for (int it: primes) {
        int c = 0;
        while (n % it == 0) {
            n /= it;
            c++;
        }
        ans *= c + 1;
        num *= it;
    }
    int all = 0;
    for (int i = 1; i < num; i++) {
        bool can = 1;
        for (int it: primes) {
            if (i % it == 0) {
                can = 0;
            }
        }
        if (can) {
            ll o = i;
            for (; o * o < n; o += num) {
                if (n % o == 0) {
                    all += 2;
                }
            }
            if (o * o == n) {
                all++;
            }
        }
    }
    return ans * all;
}

// Combinators
/* to call this namespace you have to
 * call init(2e5 + 5 , 1e7 + 9);
 * */
namespace combinatorics {
    ll MOD;
    vector<ll> fac, inv, finv;

    ll nCr(ll x, ll y) {
        if (x < 0 || y > x)return (0);
        return (fac[x] * finv[y] % MOD * finv[x - y] % MOD);
    }
    ll nPr (ll x , ll y){
        if (x < 0 || y > x || y < 0) return 0;

        return fac[x] * finv[x - y] % MOD;
    }
    ll power(ll b, ll n) {
        b %= MOD;
        ll s = 1;
        while (n) {
            if (n % 2 == 1)s = s * b % MOD;
            b = b * b % MOD;
            n /= 2;
        }
        return s;
    }

    void init(int n, ll mod) {
        fac.resize(n + 1);
        inv.resize(n + 1);
        finv.resize(n + 1);
        MOD = mod;
        fac[0] = inv[0] = inv[1] = finv[0] = finv[1] = 1;
        for (ll i = 1; i <= n; ++i)fac[i] = fac[i - 1] * i % MOD;
        for (ll i = 2; i <= n; ++i)inv[i] = MOD - MOD / i * inv[MOD % i] % MOD;
        for (ll i = 2; i <= n; ++i)finv[i] = finv[i - 1] * inv[i] % MOD;
    }

    ll Inv(int x) {
        return power(x, MOD - 2);
    }

    ll catalan(int n) {
        return (nCr(2 * n, n) * Inv(n + 1)) % MOD;
    }
};
using namespace combinatorics;


//  max number that __int128 fast_power(2 , 128) - 1
__int128 read() {
    __int128 x = 0, f = 1;
    char ch = getchar();
    while (ch < '0' || ch > '9') {
        if (ch == '-') f = -1;
        ch = getchar();
    }
    while (ch >= '0' && ch <= '9') {
        x = x * 10 + ch - '0';
        ch = getchar();
    }
    return x * f;
}
void print(__int128 x) {
    if (x < 0) {
        putchar('-');
        x = -x;
    }
    if (x > 9) print(x / 10);
    putchar(x % 10 + '0');
}
bool cmp(__int128 x, __int128 y) { return x > y; }

// function to calc if the number is prime or not max number is 1e18

ll mulmod(ll a, ll b, ll m) {
    ll res = 0;
    while (b > 0) {
        if (b & 1) res = ((m - res) > a) ? res + a : res + a - m;
        b >>= 1;
        if (b) a = ((m - a) > a) ? a + a : a + a - m;
    }
    return res;
}

ll power(ll a, ll b, ll m) {
    if (b == 0) return 1;
    if (b & 1) return mulmod(a, power(a, b - 1, m), m);
    ll tmp = power(a, b / 2, m);
    return mulmod(tmp, tmp, m);
}

bool Prime(ll n) {
    if (n <= 1)return 0;
    for (int i = 0; i < 10; i++) {
        ll tmp = (rand() % (n - 1)) + 1;
        if (power(tmp, n - 1, n) != 1)
            return false;
    }
    return true;
}
// function to know each prime factors in a short time U can optimize

const int N4 = 1e5 + 5;
int p[N4];
void seiveProcessing (){
    for (int i = 2; i < N; i++){
        if (p[i] == 0)
        for (int j = i; j < N; j += i){
            p[j] ++;
        }
    }
}
/*
signed main() {
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    ll n;
    cin >> n;
    if (prime(n)) {
        cout << "Prime";
    } else {
        cout << "Not Prime";
    }
    return 0;
}
 */
/// The End of Number theory
