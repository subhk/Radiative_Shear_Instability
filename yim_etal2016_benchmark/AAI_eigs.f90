program driver

  implicit none
  !!!!!!!!!!!!!!!!! Feast declaration variable
  integer,dimension(64) :: fpm 
  integer :: loop
  !!!!!!!!!!!!!!!!! Matrix declaration variable
  character(len=100) :: name
  integer :: n1, n2, n, nnz
  double precision :: rea, img
  integer,dimension(:),allocatable :: isa,jsa,isb,jsb,ic,jc
  complex(kind=kind(1.0d0)),dimension(:),allocatable :: c, sa, sb
  double precision,dimension(:),allocatable :: cc

  !!!!!!!!!!!!!!!!! Others
  integer :: i,k, kk
  integer :: M0,M,info
  integer, dimension(:), allocatable :: max_loc
  complex(kind=(kind(1.0d0))) :: Emid, eigval
  double precision :: r,epsout
  complex(kind=(kind(1.0d0))),dimension(:),allocatable :: E ! eigenvectors
  complex(kind=(kind(1.0d0))),dimension(:,:),allocatable :: X ! eigenvectors
  double precision,dimension(:),allocatable :: res ! eigenvalue+residual
  character(len=5) :: charI

  character(len=40) :: fmt ! format descriptor
  character(len=1) :: xx1
  character(len=2) :: xx2

  !fmt = '(I0.0)' ! an integer of width 5 with zeros at the left
    

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!Read Coordinate format and convert to csr format
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  do kk=2, 2

    !!! A matrix 
    if (kk < 10) then
        write (xx1, '(I0.0)') kk
        open(10, file = 'systemA_'//trim(xx1)//'.mtx', status='old')
    else
        write (xx2, '(I2.0)') kk
        open(10, file = 'systemA_'//trim(xx2)//'.mtx', status='old')
    endif
    print *, kk

    read(10,*)
    read(10,*) n,n,nnz
    allocate(ic(nnz))
    allocate(jc(nnz))
    allocate(c(nnz))
    do i=1, nnz
        read(10,*) ic(i), jc(i), rea, img
        c(i) = rea*(1.0d0,0.0d0) + img*(0.0d0,1.0d0)
    end do
    close(10)

    if (kk==2) then
      allocate(isa(1:n+1))
      allocate(jsa(1:nnz))
      allocate(sa(1:nnz))
    endif
    call zcoo2csr(n,nnz,ic,jc,c,isa,jsa,sa)
    deallocate(ic,jc,c)

  ! !!! B matrix 
    if (kk < 10) then
        open(10, file = 'systemB_'//trim(xx1)//'.mtx', status='old')
      else
        open(10, file = 'systemB_'//trim(xx2)//'.mtx', status='old')
    endif
    read(10,*)
    read(10,*) n,n,nnz
    allocate(ic(nnz))
    allocate(jc(nnz))
    allocate(c(nnz))
    do i=1, nnz
        read(10,*) ic(i), jc(i), rea !, img 
        c(i) = rea*(1.0d0,0.0d0) + 0d0*(0.0d0,1.0d0)
    end do
    close(10)

    if (kk==2) then
      allocate(isb(1:n+1))
      allocate(jsb(1:nnz))
      allocate(sb(1:nnz))
    endif
    call zcoo2csr(n,nnz,ic,jc,c,isb,jsb,sb)
    deallocate(ic,jc,c)

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!! FEAST in sparse format !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Emid = (0.25d0, -0.25d0)
    r    = 0.25d0 

    M0 = 30 !! M0>=M

  !!!!!!!!!!!!! ALLOCATE VARIABLE 
    if (kk==2) then
      allocate(E(1:M0))       ! Eigenvalue
      allocate(X(1:N,1:2*M0)) ! Eigenvectors
      allocate(res(1:2*M0))   ! Residual 
    endif

  !!!!!!!!!!!!!  FEAST
    call feastinit(fpm)
    fpm(1)  = 1     ! Print: on-screen (1), off-screen (0)
    fpm(3)  = 6     ! Tolerance : 1e-fpm(3)
    fpm(4)  = 100   ! Max number of iterations
    fpm(8)  = 200   ! Number of contour points
    fpm(10) = 1     ! Store linear system factorization in memory
    fpm(16) = 1     ! Integration type: Gauss(0), Trapezoidal (1)
    fpm(42) = 1     ! Linear solver: single-precision (1), double-precision (0) 
    call zfeast_gcsrgv(N,sa,isa,jsa,sb,isb,jsb,fpm,epsout,loop,Emid,r,M0,E,X,M,res,info)

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!! POST-PROCESSING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print *,'FEAST OUTPUT INFO', info
    if (info/=0) print *,'zfeast_gcsrgv -- failed'
    if (info==0) then
      print *,'zfeast_gcsrgv -- success'
      print *,'*************************************************'
      print *,'************** REPORT ***************************'
      print *,'*************************************************'
      print *,'Eigenvalues/Residuals (inside interval)'
      do i=1, M
        print *, i, E(i), res(i)
      enddo

      max_loc = findloc(real(E(1:M)), maxval(real(E(1:M))))
      eigval  = real(E(max_loc(1)))*(1.0d0,0.0d0) + aimag(E(max_loc(1)))*(0.0d0,1.0d0)

      print *, '*************************************************'
      print *, 'Maximum Growth rate (inside interval)'
      print *,  kk, eigval
      print *, '*************************************************'

      if (kk < 10) then
        open(10, file = 'real_eigenvec_'//trim(xx1)//'.dat', status='new')
        open(20, file = 'imag_eigenvec_'//trim(xx1)//'.dat', status='new')
      else
        open(10, file = 'real_eigenvec_'//trim(xx2)//'.dat', status='new')
        open(20, file = 'imag_eigenvec_'//trim(xx2)//'.dat', status='new')
      endif
      do i=1,N
          write(10, *) real(X(i,max_loc(1)))
          write(20, *) aimag(X(i,max_loc(1)))  
      end do  
      close(10)
      close(20) 
    endif

  enddo

end program driver



