
// segment_tree code
// pure
#include <bits/stdc++.h>
using namespace std;
#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()
#define int  long long
#define lll __int128
//#define ceil(n, m) (((n) / (m)) + ((n) % (m) ? 1 : 0))
#define endl '\n'
const long long INF = 1ll << 32;
const long double PI = acos(-1);
const int N = 1000001, Mod = 1000000007 , inf = 1e9 , bits = 27;
const int NN = 1e9 , OO = 0x3F3F3F3F;
#define NeedForSpeed ios_base::sync_with_stdio(false) , cin.tie(nullptr),cout.tie(nullptr);
struct SegmentTree {
    int size;
    vector <int> seg;
    void init (int  n ){
        size = 1;
        while (size < n ) size *= 2;
        seg.assign(size * 2 , 0);
    }

    void set (int i , int v , int x , int lx , int rx){
        if (rx - lx == 1){
            seg[x] = v;
            return ;
        }
        int m = (lx + rx ) / 2;
        if (i < m){
            set(i , v , x * 2 + 1 ,lx , m);
        }
        else {
            set (i , v , x * 2 + 2 , m , rx);
        }
        seg[x] = seg [x * 2 + 1] + seg[x * 2 + 2];
    }
    void set (int i , int v){
        set (i , v , 0 , 0 , size);
    }
    int calc (int l , int r , int x , int lx , int rx){
        if (l >= rx || r <= lx ) return 0;
        else if (lx >= l && rx <= r) return seg[x];
        else {
            int m = (lx + rx ) / 2;
            int s1 = calc(l , r , x * 2 + 1 , lx , m);
            int s2 = calc(l , r , x * 2 + 2 , m , rx);
            return s1 + s2;
        }
    }
    int calc (int l , int r ){
        return calc(l , r , 0 , 0 , size);
    }
};
void solve1 (){
    int n , m;
    cin >> n >> m;
    SegmentTree st;
    st.init (n);
    for (int i = 0;i < n;i++){
        int v;
        cin >> v;
        st.set(i , v);
    }
    while (m--){
        int ty;
        cin >> ty;
        if (ty == 1){
            int i , v;
            cin >> i >> v;
            st.set (i , v);
        }
        else {
            int l , r;
            cin >> l >> r;
            cout << st.calc(l , r ) << endl;
        }
    }
}
// ************
// another way to code segment tree

struct SegmentTree1 {
private :
    int sz ;
    vector <int> seg;
    int op ( int a , int b){
        return a + b;
    }
    void update (int i , int v , int x ,int lx , int rx){
        if (lx == rx){
            seg[x] = v;
            return;
        }
        int m = (lx + rx) / 2;
        if (i <= m){
            update (i , v , x * 2 + 1 , lx , m);

        }
        else update (i , v , x * 2 + 2 , m + 1 , rx);
        seg[x] = op(seg[x * 2 + 1 ] , seg[x * 2 + 2 ]);
    }
    int query (int l , int r , int x , int lx , int rx){
        if ( r < lx || rx < l) return 0;
        else if (l <= lx && rx <= r) return seg[x];
        else {
            int m = (lx + rx) / 2;
            return op(
            query(l , r , x * 2 + 1 , lx , m),
            query(l , r , x * 2 + 2 , m + 1 , rx)
            );
        }
    }
public:
    SegmentTree1(int n ){
        sz = 1;
        while (sz < n)sz <<= 1;
        seg = vector<int>(sz << 1);

    }
    void update (int i , int v){
        update (i , v , 0  , 0 , sz -1);
    }
    int query (int l , int r){
        return query(l , r , 0 , 0 , sz-1);
    }
};

[[noreturn]] void solve (){
   int n , m;
   cin >> n >> m;
   SegmentTree1 seg(n);
   for (int i = 0;i < n;i++){
       int x;
       cin >> x;
       seg.update(i , x);
   }
   while (m--) {
       int type, l, r;
       cin >> type >> l >> r;

       if (type == 1) {
           seg.update(l, r);
       } else cout << seg.query(l, r - 1) << endl;
   }
}
int32_t main() {
    NeedForSpeed
    int test_cases = 1;
   // cin >> test_cases;
    while (test_cases--) {
        solve ();
    }
    return 0;
}
