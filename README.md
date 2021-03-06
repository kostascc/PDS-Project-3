## Εργασία 3 - Παράλληλα & Διανεμημένα Συστήματα

## Χρόνοι σε Tesla P100 GPU

|         | 64x64 | 128x128 | 256x256 | 340x340 | 512x512 |
| ------- | ----- | ------- | ------- | ------- | ------- |
| [seconds] | 0.01  |  0.3    |  6.1    |   8.6   |  108.9  |



## Εκκίνηση σε Google Colab
Μπορείτε να τρέξετε τον κώδικα σε [Google Colab](https://colab.research.google.com/drive/1eqEE-xPxuQXFzxg1lUrESmFQ3nTqIC8B?usp=sharing) απευθείας.



## Εκκίνηση Τοπικά

Για δοκιμή του κώδικα δεν χρειάζεται εγκατάσταση του Visual Studio. Μπορεί να γίνει απλά με τον nvcc:

````
# Get in the source directory
cd NLM

# Error regarding file name from Visual Studio, not recognized by nvcc
cp parameters.h Parameters.h 

# Compile in nvcc with OpenMP
nvcc -o main.o main.cu CPU.cpp GPU.cu Utils.cpp -Xcompiler -fopenmp

# Execute
./main.o -gpu -i ../input/rectangular_128x128_n4.bmp -o ../output/
````


Τα Arguments που μπορούν να χρησιμοποιηθούν είναι:

````
 -cpu                 : Run CPU algorithm
 -gpu                 : Run GPU algorithm
 -t <int>             : Set CPU threads (Default: 8)
 -i ./<dir>/<img>.bmp : Input Image (Default: ../input/rectangular_128x128_n4.bmp)
 -o ./<out_dir>       : Output Directory (../output)
 -s <float>           : Sigma (Default: 0.13)
````
Για Visual Studio, έχουν προστεθεί αντίστοιχα Debugging Configurations με εικόνες 128x128 και 256x256.

