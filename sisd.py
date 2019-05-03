import numpy as np
A = np.zeros((100, 100))
for i in range(100): #generate A
    for j in range(100):
        if (i == j):
            A[i, j] = 2
        if (abs(i - j) == 1):
            A[i, j] = A[j, i] = -1
b = np.ones((100, 1))  #generate b
# print(A)
# print(b)
print("Conjugate Gradient x:")
x=np.zeros((100,1)) # 初始值x0
r=b-np.dot(A,x)
p=r  #p0=r0
#while np.linalg.norm(np.dot(A, x) - b) / np.linalg.norm(b) >= 10 ** -6:
for i in range(10):
    r1=r
    a=np.dot(r.T,r)/np.dot(p.T,np.dot(A,p))
    x = x + a * p    #x(k+1)=x(k)+a(k)*p(k)
    r=b-np.dot(A,x)  #r(k+1)=b-A*x(k+1)
    q = np.linalg.norm(np.dot(A, x) - b) / np.linalg.norm(b)
    if q<10**-6:
        print()
        break
    else:
        beta=np.linalg.norm(r)**2/np.linalg.norm(r1)**2
        p=r+beta*p  #p(k+1)=r(k+1)+beta(k)*p(k)
 
print(x)
print("done Conjugate Gradient!")
