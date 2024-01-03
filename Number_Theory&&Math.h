#include <bits/stdc++.h>
using namespace std;
void  NeedForSpeed(){
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    freopen("input.txt","r",stdin);
    freopen("output.txt","w",stdout);
}
#define int  long long
#define ll long long
#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()
#define Create_File freopen("jenga.in", "r", stdin)
#define endl '\n'
#define sz(s)	(int)(s.size())
const long long INF = 1ll << 32;
const long double PI = acos(-1);
const int N = 200005, Mod = 1000000007;
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
int fast_power(int n , int p ){
    int result = 1;
    while (p > 0){
        if (p % 2){
            result = (result  *  n);
    }
        n = ((n ) * (n ));
        p >>= 1;
    }
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
bool isPrime(int n)
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
int32_t main (){
    NeedForSpeed();
    modified_sieve();
}
