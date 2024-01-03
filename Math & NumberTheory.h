#include <bits/stdc++.h>

using namespace std;
using namespace std;
using int64 = int64_t;
#define int    long long
#define lll __int128
#define ordered_set tree<int, null_type,less<int>, rb_tree_tag,tree_order_statistics_node_update>
#define Ceil(n, m) (((n) / (m)) + ((n) % (m) ? 1 : 0))
#define endl '\n'
#define NeedForSpeed ios_base::sync_with_stdio(false) , cin.tie(nullptr), cout.tie(nullptr);
const int64 INF = 1000000000LL + 100;
const long double PI = acos(-1) , EPS = 0.000000001;
const int N1 = 2e5 + 7, Mod = 1000000007 , inf = 1e9 , bitstr = 27;
const int N2 = 1e5+5;
using int64 = int64_t;
#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbeg    in(), (x).rend()
const int NN = 7000000 , OO = 0x3F3F3F3F;
// sum of odd numbers
//
 int dx[]={0,0,1,-1,-1,-1,1,1};
//// Delta Y array is the step of vertical step
int dy[]={1,-1,0,0,1,-1,1,-1};

int sum_odd (int n ){
    int ans = (n + 1) / 2;
    return ans * ans;
}
// to convert binary string to decimal integer
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
*//**/*/*/*/

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
int fast_power(int n , int p ){
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
            prec_n(max(prec_n, 3)),
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
int32_t main (){
    NeedForSpeed;
    modified_sieve();
}
