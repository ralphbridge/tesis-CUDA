PROGRAM kdfortran
        IMPLICIT NONE

        INTEGER :: N,Nk,Ne ! Number of particles (N), number of modes (Nk), number of polarizations (Ne)

        REAL(8),ALLOCATABLE :: init(:) ! Initial positions
        REAL(8),ALLOCATABLE :: pos(:) ! Record of final positions

        REAL(8),ALLOCATABLE :: eta(:) ! Random phases (2N)

        REAL(8),ALLOCATABLE :: theta(:)
        REAL(8),ALLOCATABLE :: phi(:)
        REAL(8),ALLOCATABLE :: k(:)
        REAL(8),ALLOCATABLE :: xi(:) ! Polarization angles

        logical :: e ! To check if files exist
        REAL(8) :: start,finish ! Variables to record total time

        ALLOCATE(pos(N))
        ALLOCATE(init(N))

        ALLOCATE(eta(2*N))

        ALLOCATE(theta(Nk))
        ALLOCATE(phi(Nk))
        ALLOCATE(k(Nk))
        ALLOCATE(xi(Nk))

        1 FORMAT(E18.10) ! Must be a 8-digit difference
        2 FORMAT(E30.22,',',E18.10,',',E18.10,',',E18.10,';')

        CALL CPU_TIME(start) ! This part records execution time 
        INQUIRE(FILE='input.txt',EXIST=e)
        IF (e) THEN
                OPEN(20,FILE="input.txt",STATUS="old")
                CLOSE(20,STATUS="delete")
        END IF
        INQUIRE(FILE='k-vectors.txt',EXIST=e)
        IF (e) THEN
                OPEN(19,FILE="k-vectors.txt",STATUS="old")
                CLOSE(19,STATUS="delete")
        END IF
        INQUIRE(FILE='screen.txt',EXIST=e)
        IF (e) THEN
                OPEN(18,FILE="screen.txt",STATUS="old")
                CLOSE(18,STATUS="delete")
        END IF
        INQUIRE(FILE='angles.txt',EXIST=e)
        IF (e) THEN
                OPEN(17,FILE="angles.txt",STATUS="old")
                CLOSE(17,STATUS="delete")
        END IF

        CALL kdeffect(N,Nk,Ne,init,pos,eta,theta,phi,k,xi) ! External C++ function in kdcuda.cu

        OPEN(3,FILE="input.txt",STATUS="new")
        WRITE(3,1) init
        CLOSE(3)

        OPEN(4,FILE="screen.txt",STATUS="new")
        WRITE(4,1) pos
        CLOSE(4)

        OPEN(5,FILE="angles.txt",STATUS="new")
        WRITE(5,1) eta
        CLOSE(5)

        OPEN(6,FILE="k-vectors.txt",STATUS="new")
        DO i=1,Nk
                WRITE(6,2) k(i),theta(i),phi(i),xi(i)
        END DO
        CLOSE(6)
        CALL CPU_TIME(finish)

        PRINT*,"Tiempo total:",finish-start,"s"

END PROGRAM
