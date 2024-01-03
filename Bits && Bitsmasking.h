#include <bits/stdc++.h>
using namespace std;
#define int  long long
#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()
#define Create_File freopen("jenga.in", "r", stdin)
#define endl '\n'
const long long INF = 1ll << 32;
const long double PI = acos(-1);
const int N = 200005, Mod = 1000000007;
const int NN = 1e6 + 7 ;
#define NeedForSpeed ios_base::sync_with_stdio(false) , cin.tie(nullptr), cout.tie(nullptr);

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
int32_t main() {
    NeedForSpeed
    int test_cases = 1;
    //cin >> test_cases;
    int cases = 1;
    while(test_cases --) {

    }
    return 0;
}