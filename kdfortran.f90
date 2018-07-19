PROGRAM kdfortran
	IMPLICIT NONE
	
	INTEGER :: i,N,Nk
!	INTEGER(4) :: rows
	REAL(8) :: dt,D,zimp,v0,wL,wres,Damping,Delta,E0L
	!REAL(8), DIMENSION(3*N) :: phi
	!REAL(8), DIMENSION(N,3) :: ang ! Table of randomly generated numbers (cols 1 and 2 from uniform and 3 from normal distribution)
        REAL(8), ALLOCATABLE :: theta(:)
        REAL(8), ALLOCATABLE :: phi(:)
        REAL(8), ALLOCATABLE :: k(:)
        REAL(8), ALLOCATABLE :: xi(:)
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! %%%%%%%%%%% IF YOU WANT TO RECORD THE COMPLETE POSITIONS, UNCOMMENT %%%%%%%%%%%%%%
! %%%%%%%%%%%%%%%%%%%%%%%%% MARKED BY THIS DELIMITERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	!REAL(8), DIMENSION(N) :: pos ! Vector of the impact positions for each particle
	REAL(8), ALLOCATABLE :: eta(:)
        REAL(8), ALLOCATABLE :: init(:)
        REAL(8), ALLOCATABLE :: pos(:)
	!REAL(8), ALLOCATABLE :: posy(:),posz(:) ! Complete "vector" of the positions (y,z) for each particle
	!REAL(8), ALLOCATABLE :: postab(:,:) ! Table formed from vectors "posy" and "posz"
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	LOGICAL :: e
	REAL(8) :: start,finish
	
	N=1						! Number of particles
	dt=1d-17					! Time step in seconds
	D=125d-6						! Laser beam waist in meters
	zimp=24d-2+D						! Distance from laser to screen in meters
	v0=1.1d7						! Electron velocity
	E0L=2d8						! Laser electric field
        wL=3.54d15
!       RESONANCE FREQUENCY FOR THE SHARP RESONANCE CONDITION
	wres=wL
!	wres=(9.10938356d-31)*((299792458d0)**2d0)/(1.0545718d-34) // wC ~ 7.8e20 rad/s

        Damping=6.245835d-24
	PRINT*,"Damping=",Damping,"s"
        Delta=100*Damping*(wres**2d0)
        Nk=10000
        PRINT*,"Delta=",Delta,"1/s"

        ALLOCATE(theta(Nk))
        ALLOCATE(phi(Nk))
        ALLOCATE(k(Nk))
        ALLOCATE(xi(Nk))
	
	1 FORMAT(E18.10) !Must be an 8-digit difference between the number after the E and the number of digits after the decimal dot
        2 FORMAT(E30.22,',',E18.10,',',E18.10,',',E18.10';')
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	!2 FORMAT(<2*N>E11.3) ! Trajectories
	!3 FORMAT(E18.10) ! Screen
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	PRINT*,"-------------------------------------------------------------"
	PRINT*,"N=",N,"particles"
	PRINT*,"E0L=",E0L,"V/m"
        PRINT*,"Nk=",Nk,"modes"
	PRINT*,"dt=",dt,"s"
	PRINT*,"v0=",v0,"m/s"
	PRINT*,"zscreen=",zimp,"m"

!	PRINT*,"zimp=",zimp
!	PRINT*,"v0=",v0
!	PRINT*,"dt=",dt
!	PRINT*,"v0*dt=",v0*dt
!	PRINT*,"zimp/(v0*dt)=",zimp/(v0*dt)
!	rows=DINT(zimp/(v0*dt))
!	PRINT*,"rows=",rows
	!ALLOCATE(phi(3*N))
	!ALLOCATE(ang(N,3))
        ALLOCATE(eta(2*N))
        ALLOCATE(init(N))
	ALLOCATE(pos(N))
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	!ALLOCATE(posy(rows*N)) ! Y-components for the positions
	!ALLOCATE(posz(rows*N)) ! Z-components for the positions
	!ALLOCATE(postab(rows,2*N)) ! Full table for the trajectories
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	CALL CPU_TIME(start)
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


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	CALL kernel_wrapper(init,pos,N,theta,phi,k,xi,eta,dt,D,zimp,v0,wL,wres,Delta,Nk,E0L) ! Without
	!CALL kernel_wrapper(phi,pos,posy,posz,rows,N,dt,D,zimp,v0,E0) ! With
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

!	k=1
!	DO i=1,N
!		DO j=k,k+2
!			ang(i,j-k+1)=phi(j)
!		END DO
!		k=j
!	END DO
	
	OPEN(20,FILE="input.txt",STATUS="new")
	!PRINT*, 'phi = ', (phi(i), i=1,3*N)
!	DO i=1,N
!		!PRINT*, 'ang = ', (ang(i,j), j=1,3)
!		WRITE(20,1) (ang(i,j), j=1,3)
!	END DO
        WRITE(20,1) init
	CLOSE(20)

        OPEN(19,FILE="k-vectors.txt",STATUS="new")
        DO i=1,Nk
                WRITE(19,2) k(i),theta(i),phi(i),xi(i)
        END DO
        CLOSE(19)
	!PRINT*,"yi=",ang(1,3)
	!PRINT*,"yf=",pos(1)
	
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	!DO j=1,N
	!	DO i=1,rows
	!		postab(i,2*j-1)=posy((j-1)*rows+i)
	!		postab(i,2*j)=posz((j-1)*rows+i)
	!	END DO
	!END DO

	!OPEN(19,FILE="paths.txt",STATUS="new")
	!DO i=1,rows
	!	WRITE(19,2) (postab(i,j), j=1,2*N) ! Trajectories
	!END DO
	!CLOSE(19)
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	OPEN(18,FILE="screen.txt",STATUS="new")
	WRITE(18,1) pos ! Impact positions
	CLOSE(18)

        OPEN(17,FILE="angles.txt",STATUS="new")
	WRITE(17,1) eta
        CLOSE(17)
	CALL CPU_TIME(finish)
	
	PRINT*,"Tiempo total:",finish-start,"s"

END PROGRAM
