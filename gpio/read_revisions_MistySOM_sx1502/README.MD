i2ctest
=======

  i2ctest - Testing program for validation of Smarc RZ/G2L or MistySOM RZ/G2L i2c hardware interfaces to onboard SoC system-on-chips including:

- Renesas pmic raa215300   
- TI tlv320aic23b              
- Semtech SX150x GPIO port expander (for MistySOM)      
- Semtech SX150x GPIO port expander (for MistyCarrier)
- Omnivision video camera sensor ov5645 (used by Coral Camera)

BUILDING
--------

i2ctest is a pretty standard autotools project. It does not depend on any
libraries other than the standard C library with GNU extensions.
The autoconf version needed to compile the project is 2.61.

Steps to Build i2ctest program

1. Build the RZ/G2L or RZ/V2L BSP and SDK. 

    `$ cd ${WORK}/build`  
    `$ bitbake core-image-weston -c populate_sdk`

    When building finished, you can get the core-image-weston SDK at 
    tmp/deploy/sdk/poky-glibc-64-core-image-weston-aarch64-smarc-rzg2l-toolchain-3.1.14.sh

2. Install the SDK, just run:

    `$ sudo {SDK_DIR}/poky-glibc-x86_64-core-image-weston-aarch64-smarc-rzg2l-toolchain-3.1.14.sh`

    `Poky (Yocto Project Reference Distro) SDK installer version 3.1.14`

    `Enter target directory for SDK (default: /opt/poky/3.1.14):`  
    `The directory "/opt/poky/3.1.14" already contains a SDK for this architecture.`  
    `If you continue, existing files will be overwritten! Proceed [y/N]? y`  
    `[sudo] password for::`   
    `Extracting   SDK.............................................................done`  
    `Setting it up...done`  
    `SDK has been successfully set up and is ready to be used.`

3. Source the cross-toolchain envirionment setup file. Installing the cross-toolchain creates a cross-toolchain environment setup script in the directory that the SDK was installed.  

    `$ source /opt/poky/3.1.14/environment-setup-aarch64-poky-linux`

4. Create the configure Script: Use the autoreconf command to generate the configure script:

    `$ autoreconf`

     The autoreconf tool takes care of running the other Autotools such as aclocal, autoconf, and automake.

5. Making the makefile: This command creates the makefile which will  
    compile the project using the cross-compiler.  
    The --disable-dependency-tracking flag disables usage of the depcomp dependency tracking by autotools configure
    (since the aarch64 compiler is unknown to depcomp, the autoconf dependency tracking feature must be
    disabled).  
    The CONFIGURE_FLAGS environment variable provides all the other minimal arguments for GNU configure:  

    `$ ./configure --disable-dependency-tracking ${CONFIGURE_FLAGS}`  

6. Make and Install the Project: These two commands compile the i2ctest program and install it into the destination directory:

    `$ make`  
    `$ make install DESTDIR=./tmp`  

7. Verify that i2c.c compiled to i2c.o properly, and that the link step created the i2ctest executable.

