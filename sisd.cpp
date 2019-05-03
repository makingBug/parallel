#include<iostream>
#include<vector>
#include<cmath>
#include<stdio.h>
using namespace std;
int N =10;
//初始化A为一个N*N的矩阵的对称矩阵
vector<vector<double> > A(N,vector<double>(N,0));

//初始化数组b
vector<double> b(N,1);

 //初始化残差r,结果x,计算方向向量d
vector<double> r(N,-1);
vector<double> d(N,0);
vector<double> x(N,0);

void displayA(vector<vector<double> > &a){
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            cout<<a[i][j]<<"    ";
        }
        cout<<endl;
    }
}

void displayb(vector<double> &b){
    for(int i=0;i<b.size();i++){
        cout<<b[i]<<" ";
    }
    cout<<endl;
}

//计算内积，也就是模的平方
double INNER_PRODUCT(vector<double> &a,vector<double>&b){
    double res = 0;
    for(int i=0;i<N;i++){
        res+=a[i]*b[i];
    }
    return res;
}

//更新残差 r = A*x-b
vector<double>&  MATRIX_VECTOR_PRODUCT(vector<vector<double> > &a,vector<double> &x,vector<double>& b){
    double temp = 0;
    for(int i=0;i<N;i++){
        temp = 0;
        for(int j=0;j<N;j++){
            temp += a[i][j]*x[j];
        }
        r[i] = temp - b[i];
    }
    return r;
}

//计算dtAd
double MATRIX_PRODUCT(vector<vector<double> >&a,vector<double> &d){
    double res = 0;
    double temp = 0;
    for(int i=0;i<N;i++){
        temp = 0;
        for(int j = 0;j<N;j++){
            temp += d[j]*A[i][j];
        }
        res += temp*d[i];
    }
    return res;
}


int main(){

    //初始化A
    for(int i=0;i<N;i++){
        for(int j =0;j<N;j++){
            if(i==j){
                A[i][j] = 2;
            }
            if(abs(i-j) == 1){
                A[i][j] = -1;
            }
        }
    }
    //displayA(A);
    //displayb(b);
    //开始迭代
    int count = 0;
    for(int i =0;i<N;i++){
        count++;

        //计算r^Tr,
        double denom1 = INNER_PRODUCT(r,r);
        r = MATRIX_VECTOR_PRODUCT(A,x,b);
    
        double num1 = INNER_PRODUCT(r,r);
        if(num1 < 0.000001){
            break;
        }
        double temp = num1/denom1;
        //计算方向向量d
        for(int j = 0;j<N;j++){
            d[j] = -r[j]+temp*d[j];
        }
        double num2 = INNER_PRODUCT(d,r);
        double denom2 = MATRIX_PRODUCT(A,d);

        //计算步长
        double  length = -num2/denom2;

        //修正x
        for(int j=0;j<N;j++){
            x[j] = x[j]+ length*d[j];
        }
    }

    cout<<"迭代次数: "<<count<<endl;
    displayb(x);
    return 0;
}
