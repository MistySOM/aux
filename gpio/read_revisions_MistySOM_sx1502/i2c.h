/*
 * i2c.h
 *
 * @date 2022/10/06
 * @author Gord Finlay
 */

#include <stdint.h>

#ifndef SRC_I2C_H_
#define SRC_I2C_H_

/*
 * Configuration for the I2C device.
 */
struct I2cDevice {
	char* filename; /**< Path of the I2C bus, eg: /dev/i2c-0 */
	uint16_t addr; /**< Address of the I2C slave, eg: 0x48 */

	int fd; /**< File descriptor for the I2C bus */
};

int main(void);

#endif /* SRC_I2C_H_ */
