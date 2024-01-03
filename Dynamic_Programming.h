#include <bits/stdc++.h>
using namespace std;
#define int  long long
const int MAX = 1e5 + 7;
int savedAnswer[MAX];
int fib (int n ){
    if (n <= 1)
        return 1;
    return fib(n -1) + fib (n - 2);
}
int fabSave(int n) {
    if (n <= 1)
        return 1;
    if (savedAnswer[n] != -1)
        return savedAnswer[n];
    return savedAnswer[n] = fabSave(n - 2) + fabSave(n - 1);
}

int pure_dp_fib (int n){
    int fib[MAX] ;
    fib[0] = fib[1] = 1;
    for (int i = 2;i <= n;i++){
        fib[i] = fib[i-1] + fib[i-2];
    }
    return fib[n];
}
//// Vacation
/// https://v...content-available-to-author-only...e.net/problem/AtCoder-dp_c

int mem[3][N];
int dp(int i, int prevAction) {

    if (i == n)return 0;

    if (prevAction != 3 and mem[prevAction][i])
        return mem[prevAction][i];

    int ret = 0;
    for (int j = 0; j < 3; ++j) {
        if (prevAction != j) {
            ret = max(
                    ret,
                    dp(i + 1, j) + arr[j][i]
            );
        }
    }
    if (prevAction != 3)
        mem[prevAction][i] = ret;
    return ret;
}
int dp() {
    stack<state> st;
    state curState;
    curState.i = 0, curState.ls = 3, curState.done = false;
    st.push(curState);
    int i, ls;
    while (st.size()) {
        curState = st.top();
        st.pop();
        i = curState.i, ls = curState.ls;
        if (i == n) {
            mem[i][ls] = 0;
        } else if (mem[i][ls] == -1) {
            if (curState.done) {
                for (int j = 0; j < 3; ++j) {
                    if (ls == j)continue;
                    mem[i][ls] = max(
                            mem[i + 1][j] + arr[i][j],
                            mem[i][ls]
                    );
                }
            } else {
                curState.done = true;
                st.push(curState);
                for (int j = 0; j < 3; ++j) {
                    if (ls == j) {
                        curState.i = i + 1;
                        curState.ls = j;
                        curState.done = false;
                        st.push(curState);
                    }
                }
            }
        }
    }
    return mem[0][3];
}
int32_t main (){
    memset (savedAnswer , -1 , sizeof savedAnswer);

    cout << fabSave(50) << endl;
return 0;
}