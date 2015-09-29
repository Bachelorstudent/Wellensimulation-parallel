# Wellensimulation-parallel

Dies ist die Wellensimulation mit Parallelisierung.
Es wurden keine experimentellen oder halbfertige Funktionen hochgeladen.
Dieser Code wurde in der Bachelorarbeit zum ermitteln der Simulationsgeschwindigkeiten verwendet.

Kompilierung:
```
mpicc -Wall -Wextra -Werror -O3 -march=native -funroll-loops -c -o wavestate3d.o -lm wavestate3d.c
mpicc -Wall -Wextra -Werror -O3 -march=native -funroll-loops -c -o wave3d.o -lm wave3d.c
mpicc wave3d.o wavestate3d.o -o wave3d -lm
````
Ausführung:
```
mpirun -np prozesse wave3d nx ny nz stepweite iterations xQuader yQuader zQuader
```
