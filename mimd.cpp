#include<iostream>
#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include<cmath>
#include<string>
#include<vector>
using namespace std;
int N = 100;
int tag = 100;
//初始化A为一个N*N的矩阵的对称矩阵
vector<vector<double> > A(N,vector<double>(N,0));

//初始化数组b
vector<double> b(N,1);

 //初始化残差r,结果x,计算方向向量d
vector<double> r(N,-1);
vector<double> d(N,0);
vector<double> x(N,0);

//计算内积，也就是模的平方
double INNER_PRODUCT(vector<double> &a,vector<double>&b,int myid,int ThreadSize){
    double res = 0;
    for(int i=myid;i<N;i=i+ThreadSize){
        res+=a[i]*b[i];
    }
    return res;
}

//更新残差 r = A*x-b
vector<double>&  MATRIX_VECTOR_PRODUCT(vector<vector<double> > &a,vector<double> &x,vector<double>& b,int myid,int ThreadSize){
    
    double temp = 0;
    for(int i=myid;i<N;i=i+ThreadSize){
        temp = 0;
        for(int j=0;j<N;j++){
            temp += a[i][j]*x[j];
        }
        r[i] = temp - b[i];
    }
    return r;
}

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

int main(int argc, char *argv[]){
    vector<double> v(5,1);
    int ThreadSize,myid;
    MPI_Status status;
    double receive,send;
    int receiveId;
    double startwtime,endwtime;
    double sum_denom1 = 0;
    double denom1 = 0;
    double sum_num1 = 0;
    double num1 = 0;
    int iterator = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ThreadSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    //所有线程初始化矩阵A,    
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
    if (myid == 0)
        startwtime = MPI_Wtime();
    

    //2.每个线程都迭代N次数
    for(iterator = 0;iterator<N;iterator++){
        //计算内积，只计算自己的那一部分
        denom1 = INNER_PRODUCT(r,r,myid,ThreadSize);
        
        //计算残差,也只计算自己的那一部分
        r = MATRIX_VECTOR_PRODUCT(A,x,b,myid,ThreadSize);

        //计算内积，只计算自己的那一部分
        
        num1 = INNER_PRODUCT(r,r,myid,ThreadSize); 
        
        //将各个num1变量加和到进程0 sum_num1，为了算接下来的步长
        MPI_Allreduce(&num1,&sum_num1,1,MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        //MPI_Bcast(&sum_num1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if(sum_num1 < 0.00001){
            break;
        }
        //将各个denom1变量加和到进程0 sum_denom1，为了算接下来的方向向量
        MPI_Allreduce(&denom1,&sum_denom1,1,MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        //MPI_Bcast(&sum_denom1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //计算每个进程的方向向量d
        double temp = sum_num1/sum_denom1;
    
        for(int i = myid;i<N;i=i+ThreadSize){
            d[i] = -r[i]+temp*d[i];
        }

        if(myid!=0)
        {
            for(int i =myid;i<N;i=i+ThreadSize){
                //发送的地址，个数，类型，目的进程号，标记，通信子
                MPI_Send(&d[i],1,MPI_DOUBLE,0,myid,MPI_COMM_WORLD);
            }
        }
        if(myid == 0){
            vector<int> count(ThreadSize,0);
            for(receiveId = 1;receiveId<ThreadSize;receiveId++){
                for(int i=receiveId;i<N;i=i+ThreadSize){
                    MPI_Recv(&receive,1,MPI_DOUBLE,receiveId,receiveId,MPI_COMM_WORLD,&status);
                    d[receiveId+count[receiveId]*ThreadSize] = receive;
                    count[receiveId]++;
                }
            }
        }
        MPI_Bcast(&d[0], N, MPI_DOUBLE, 0, MPI_COMM_WORLD);


        //算这之前要将d同步
        double sum_num2 = 0;
        double num2 = INNER_PRODUCT(d,r,myid,ThreadSize);

        double sum_denom2 = 0;
        double denom2 = 0;
        double tmp=0;
        for(int i = myid;i<N;i=i+ThreadSize){
            tmp = 0;
            for(int j =0;j<N;j++){
                tmp += A[i][j]*d[j];
            }
            denom2+=tmp*d[i];
        }

        MPI_Reduce(&num2,&sum_num2,1,MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&denom2,&sum_denom2,1,MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        //计算步长
        double length = 0;
        if(myid == 0){
            length = -sum_num2/sum_denom2;
        }
        MPI_Bcast(&length, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        for(int i = myid;i<N;i=i+ThreadSize){
            x[i] = x[i]+ length*d[i];
        }
        if(myid!=0)
        {
            for(int i =myid;i<N;i=i+ThreadSize){
                //发送的地址，个数，类型，目的进程号，标记，通信子
                MPI_Send(&x[i],1,MPI_DOUBLE,0,100,MPI_COMM_WORLD);
            }
        }
        if(myid == 0){
            vector<int> count(ThreadSize,0);
            for(receiveId = 1;receiveId<ThreadSize;receiveId++){
                for(int i=receiveId;i<N;i=i+ThreadSize){
                    MPI_Recv(&receive,1,MPI_DOUBLE,receiveId,100,MPI_COMM_WORLD,&status);
                    x[receiveId+count[receiveId]*ThreadSize] = receive;
                    count[receiveId]++;
                }
            }
            //displayb(x);
            endwtime = MPI_Wtime();
            //cout<<"time use : "<<endwtime-startwtime<<endl;
        }
        MPI_Bcast(&x[0], N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if(myid == 0){
        
        displayb(x);
        cout<<"time use : "<<endwtime- startwtime<<endl;
        cout<< "iterator = "<<iterator<<endl;
    }
    
    
    MPI_Finalize();
    return 0;
}