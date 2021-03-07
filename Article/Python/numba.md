# Python Can Be Faster Than C++
[[Article]](https://medium.com/swlh/python-can-be-faster-than-c-2372c627068)

---

- Python은 다재다능한 프로그래밍 언어
- 대부분의 머신러닝 문제에를 해결하는데 사용되고 있음
  - 다양한 라이브러리
  - High-level 언어
- 하지만 다른 언어보다 느려 Inference 측면에서는 C++같은 다른 언어를 사용

---

## Python vs. C++
- 소수 판별 알고리즘으로 python과 c++ 연산 속도 비교

### Python
```python
import math
from time import per_counter 
def is_prime(num):
  if num == 2:
     return True;
  if num <= 1 or not num % 2:
     return False
  for div in range(3,int(math.sqrt(num)+1),2):
     if not num % div:
        return False
 return True
def run program(N):
   for i in range(N):
      is_prime(i)
if __name__ == ‘__main__’:
   N = 10000000
   start = perf_counter()
   run_program(N)
   end = perf_counter()
   print (end — start)
```

### C++
```cpp
#include <iostream>
#include <cmath>
#include <time.h>
using namespace std;
bool isPrime(int num)
{
 if (num == 2) return true; 
 if (num <= 1 || num % 2 == 0) return false;
 double sqrt_num = sqrt(double(num));
 for (int div = 3; div <= sqrt_num; div +=2){
    if (num % div == 0) return false;
 }
 return true;
}
int main() 
{
 int N = 10000000;
 clock_t start,end;
 start = clock();
 for (int i; i < N; i++) isPrime(i);
 end = clock();
 cout << (end — start) / ((double) CLOCKS_PER_SEC);
 return 0;
}
```

### 연산 속도 결과
- Python : 80.137s
- C++ : 3.174s

### Why??
- Python은 다이나믹 언어
- 또한, 안터프리터 언어로써 Parallel Programming 미지원

---

## Numba 소개

- Python, NumPy 코드를 JIT compiler로 변환해주는 오픈소스 라이브러리

```bash
pip install numba
```

## Numba를 적용한 Python Code

```python
import math
from time import per_counter 
from numba import njit, prange

@njit(fastmath=True, cache=True)
def is_prime(num):
   if num == 2:
      return True;
   if num <= 1 or not num % 2:
      return False
   for div in range(3,int(math.sqrt(num)+1),2):
      if not num % div:
        return False
   return True

@njit(fastmath=True, cache=True,parallel=True)
def run program(N):
   for i in prange(N):
      is_prime(i)

if __name__ == ‘__main__’:
  N = 10000000
  start = perf_counter()
  run_program(N)
  end = perf_counter()
  print (end — start)
```

## 연산 속도 결과
- Python : 1.401s

Python도 C++보다 빠를 수 있다!!