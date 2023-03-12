/*
 * i2c.c
 *
 * @date 2022/10/06
 * @author Gord Finlay
 * 
 */

#include <linux/i2c-dev.h>
#include <sys/ioctl.h>

#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "i2c.h"
#include "smbus.h"

// I2C bus device names
#define RIIC0_DEVICE_FILENAME "/dev/i2c-0"
#define RIIC1_DEVICE_FILENAME "/dev/i2c-1"
#define RIIC2_DEVICE_FILENAME "/dev/i2c-2"
#define RIIC3_DEVICE_FILENAME "/dev/i2c-3"

// I2C slave device addresses
#define RENESAS_PMIC_RAA215300_MAIN_SLAVE_DEV_ADDR    (0x12)
#define RENESAS_PMIC_RAA215300_RTC_SLAVE_DEV_ADDR     (0x6F)
#define TI_TLV320AIC23B_SLAVE_DEV_ADDR                (0x1A)
#define SEMTECH_SX150X_MISTYSOM_SLAVE_DEV_ADDR        (0x20)
#define SEMTECH_SX150X_MISTYCARRIER_SLAVE_DEV_ADDR    (0x21)
#define OMNIVISION_OV5645_SLAVE_DEV_ADDR              (0x3C)
#define RENESAS_HS3001_SLAVE_DEV_ADDR                 (0x44)

// Renesas PMIC RAA215300 main port registers
#define RENESAS_PMIC_RAA215300_MAIN_REG_HW_REV        (0xF8)
// Renesas PMIC RAA215300 RTC port registers
#define RENESAS_PMIC_RAA215300_RTC_REG_RTC_SR         (0x07)
// TI TLV320AIC23B registers
#define TI_TLV320AIC23B_REG_DAI                       (0x07)
// Semtech SX150X registers
#define SEMTECH_SX150X_REG_EV_STS                     (0x09)
// Omnivision OV5645 registers
#define OMNIVISION_OV5645_REG_CHIP_ID_LOW             (0x300A)
#define OMNIVISION_OV5645_REG_CHIP_ID_HIGH            (0x300B)
// Renesas HS3001 temp sensor registers, modes, and parameters
#define RENESAS_HS3001_CMD_ENTER_PROGRAMMING_MODE     (0xA0)
#define RENESAS_HS3001_CMD_EXIT_PROGRAMMING_MODE      (0x80)
#define RENESAS_HS3001_CMD_PROGRAMMING_MODE_DELAY_US  (150)
#define RENESAS_HS3001_REG_SENSOR_ID_UPPER_16BIT_WORD (0x1E)
#define RENESAS_HS3001_REG_SENSOR_ID_LOWER_16BIT_WORD (0x1F)

uint8_t hs3001_cmd_enter_prog_mode_params[] = { 0x00, 0x00 };
uint8_t hs3001_cmd_exit_prog_mode_params[] = { 0x00, 0x00 };

int i2c_start(struct I2cDevice* dev);
int i2c_read(struct I2cDevice* dev, uint8_t *buf, size_t buf_len);
int i2c_write(struct I2cDevice* dev, uint8_t *buf, size_t buf_len);
int i2c_readn_reg(struct I2cDevice* dev, uint8_t reg, uint8_t *buf, size_t buf_len);
int i2c_writen_reg(struct I2cDevice* dev, uint8_t reg, uint8_t *buf, size_t buf_len);
uint8_t i2c_read_reg(struct I2cDevice* dev, uint8_t reg);
int i2c_write_reg(struct I2cDevice* dev, uint8_t reg, uint8_t value);
void i2c_stop(struct I2cDevice* dev);

uint8_t hs3001_enter_nvm_programming_mode(struct I2cDevice *dev);
uint8_t hs3001_exit_nvm_programming_mode(struct I2cDevice *dev);
uint32_t hs3001_read_chip_id(struct I2cDevice *dev);

/*
 * Start the I2C device.
 *
 * @param dev points to the I2C device to be started, must have filename and addr populated
 *
 * @return - 0 if the starting procedure succeeded
 *         - negative if the starting procedure failed
 */
int i2c_start(struct I2cDevice *dev)
{
   int fd;
   int rc;

   /*
    * Open the given I2C bus filename.
    */
   fd = open(dev->filename, O_RDWR);
   if (fd < 0)
   {
      rc = fd;
      printf("%s: Failed to open file %s - retcode %d\r\n", __func__, dev->filename, rc);
      goto fail_open;
   }

   /*
    * Set the given I2C slave address.
    */
   rc = ioctl(fd, I2C_SLAVE, dev->addr);
   if (rc < 0)
   {
      printf("%s: Failed to set slave address %02X - retcode %d\r\n", __func__, dev->addr, rc);
      goto fail_set_i2c_slave;
   }

   dev->fd = fd;

   return 0;

fail_set_i2c_slave:
   close(fd);
fail_open:
   return rc;
}

/*
 * Read data from a register of the I2C device.
 *
 * @param dev points to the I2C device to be read from
 * @param reg the register to read from
 * @param buf points to the start of buffer to be read into
 * @param buf_len length of the buffer to be read
 *
 * @return - number of bytes read if the read procedure succeeded
 *         - 0 if no bytes were read
 *         - negative if the read procedure failed
 */
int i2c_readn_reg(struct I2cDevice *dev, uint8_t reg, uint8_t *buf, size_t buf_len)
{
   int rc;

   rc = i2c_smbus_read_i2c_block_data(dev->fd, reg, buf_len, buf);
   if (rc <= 0)
   {
      printf("%s: Failed to read i2c register data - retcode %d\r\n", __func__, rc);
   }
   return rc;
}

/*
 * Write data to the register of the I2C device.
 *
 * @param dev points to the I2C device to be written to
 * @param reg the register to write to
 * @param buf points to the start of buffer to be written from
 * @param buf_len length of the buffer to be written
 *
 * @return - number of bytes written if the write procedure succeeded
 *         - 0 if no bytes were written
 *         - negative if the write procedure failed
 */
int i2c_writen_reg(struct I2cDevice *dev, uint8_t reg, uint8_t *buf, size_t buf_len)
{
   uint8_t *full_buf;
   int full_buf_len;
   int rc;
   int i;

   rc = i2c_smbus_write_i2c_block_data(dev->fd, reg, buf_len, buf);
   if (rc <= 0)
   {
      printf("%s: Failed to write i2c register data - retcode %d\r\n", __func__, rc);
   }
   return rc;
}

/*
 * Read value from a register of the I2C device.
 *
 * @param dev points to the I2C device to be read from
 * @param reg the register to read from
 *
 * @return - the value read from the register
 *         - 0 if the read procedure failed
 */
uint8_t i2c_read_byte_reg(struct I2cDevice *dev, uint8_t reg)
{
   uint8_t value = 0;
   i2c_readn_reg(dev, reg, &value, 1);
   return value;
}

/*
 * Write value to the register of the I2C device.
 *
 * @param dev points to the I2C device to be written to
 * @param reg the register to write to
 * @param value the value to write to the register
 *
 * @return - number of bytes written if the write procedure succeeded
 *         - 0 if no bytes were written
 *         - negative if the write procedure failed
 */
int i2c_write_byte_reg(struct I2cDevice *dev, uint8_t reg, uint8_t value)
{
   return i2c_writen_reg(dev, reg, &value, 1);
}

/*
 * Read value from a register of the I2C device.
 *
 * @param dev points to the I2C device to be read from
 * @param reg the register to read from
 *
 * @return - the value read from the register
 *         - 0 if the read procedure failed
 */
uint8_t i2c_read_word_reg(struct I2cDevice *dev, uint8_t reg)
{
   uint8_t value = 0;
   i2c_readn_reg(dev, reg, &value, 2);
   return value;
}

/*
 * Write value to the register of the I2C device.
 *
 * @param dev points to the I2C device to be written to
 * @param reg the register to write to
 * @param value the value to write to the register
 *
 * @return - number of bytes written if the write procedure succeeded
 *         - 0 if no bytes were written
 *         - negative if the write procedure failed
 */
int i2c_write_word_reg(struct I2cDevice *dev, uint8_t reg, uint8_t value)
{
   return i2c_writen_reg(dev, reg, &value, 2);
}

/*
 * Stop the I2C device.
 *
 * @param dev points to the I2C device to be stopped
 */
void i2c_stop(struct I2cDevice *dev)
{
   /*
    * Close the I2C bus file descriptor.
    */
   close(dev->fd);
}

/*
 * Write command to enter Renesas HS3001 NVM Programming Mode
 *
 * @param dev points to the I2C device to be read from
 *
 * @return - 1 if the command written to enter NVM programming mode succeeded
 *         - 0 if the command failed
 */
uint8_t hs3001_enter_nvm_programming_mode(struct I2cDevice *dev)
{
   uint8_t rc;

   rc = i2c_writen_reg(dev, RENESAS_HS3001_CMD_ENTER_PROGRAMMING_MODE,
                       &hs3001_cmd_enter_prog_mode_params[0], 2);
   /*
    * Wait for command to complete.
    */
   usleep(RENESAS_HS3001_CMD_PROGRAMMING_MODE_DELAY_US);

   if (rc <= 0)
   {
      printf("Failed Renesas HS3001 I2C command to enter Renesas HS3001 programming mode - retcode %d\r\n", rc);
   }
   else
   {
      printf("Entered Renesas HS3001 programming mode\r\n");
   }
   return rc;
}

/*
 * Write command to exit Renesas HS3001 NVM Programming Mode
 *
 * @param dev points to the I2C device to be read from
 *
 * @return - 1 if the command written to exit NVM programming mode succeeded
 *         - 0 if the command failed
 */
uint8_t hs3001_exit_nvm_programming_mode(struct I2cDevice *dev)
{
   uint8_t rc;

   rc = i2c_writen_reg(dev, RENESAS_HS3001_CMD_EXIT_PROGRAMMING_MODE,
                       &hs3001_cmd_exit_prog_mode_params[0], 2);
   /*
    * Wait for command to complete.
    */
   usleep(RENESAS_HS3001_CMD_PROGRAMMING_MODE_DELAY_US);

   if (rc <= 0)
   {
      printf("Failed Renesas HS3001 I2C command to exit programming mode - retcode %d\r\n", rc);
   }
   else
   {
      printf("Exited Renesas HS3001 programming mode\r\n");
   }
   return rc;
}

/*
 * Read the Renesas HS3001 Chip ID
 *
 * @param dev points to the I2C device to be read from
 *
 * @return - 32-bit Renesas HS3001 Sensor Chip ID
 */

uint32_t hs3001_read_chip_id(struct I2cDevice *dev)
{
   int rc;
   uint8_t hs3001_chip_id_upper_word[2] = { 0 };
   uint8_t hs3001_chip_id_lower_word[2] = { 0 };

   rc = i2c_readn_reg(dev, RENESAS_HS3001_REG_SENSOR_ID_UPPER_16BIT_WORD,
                      hs3001_chip_id_upper_word, 2);
   if (rc <= 0)
   {
      printf("Failed to read upper word of Renesas HS3001 sensor ID - retcode %d\r\n", rc);
      return rc;
   }
   rc = i2c_readn_reg(dev, RENESAS_HS3001_REG_SENSOR_ID_LOWER_16BIT_WORD,
                      hs3001_chip_id_lower_word, 2);
   if (rc <= 0)
   {
      printf("Failed to read lower word of Renesas HS3001 sensor ID - retcode %d\r\n", rc);
      return rc;
   }

   return ((uint32_t)(hs3001_chip_id_upper_word[0] << 16 | hs3001_chip_id_lower_word[0]));
}

int main(void)
{
   struct I2cDevice dev_raa215300_main;
   struct I2cDevice dev_raa215300_rtc;
   struct I2cDevice dev_tlv320aic23b;
   struct I2cDevice dev_sx150x_MistySom;
   struct I2cDevice dev_sx150x_MistyCarrier;
   struct I2cDevice dev_ov5645;
//   struct I2cDevice dev_hs3001;

   int rc;

   uint8_t raa215300_hw_rev = { 0 };
   uint8_t raa215300_rtc_sr = { 0 };
   uint8_t tlv320aic23b_dai = { 0 };
   uint8_t sx150x_MistySOM_ev_sts = { 0 };
   uint8_t sx150x_MistyCarrier_ev_sts = { 0 };
   uint8_t ov5645_chip_id_low = { 0 };
   uint8_t ov5645_chip_id_high = { 0 };

   //=============================================================
   // Renesas PMIC RAA215300 main port
   //=============================================================
   //
   // Set the I2C bus filename and slave address
   dev_raa215300_main.filename = RIIC3_DEVICE_FILENAME;
   dev_raa215300_main.addr = RENESAS_PMIC_RAA215300_MAIN_SLAVE_DEV_ADDR;
   // Start the PMIC device on the main port.
   rc = i2c_start(&dev_raa215300_main);
   if (rc < 0)
   {
      printf("Failed to start Renesas PMIC i2c device main port - retcode %d\r\n", rc);
   }
   else
   {
      rc = i2c_readn_reg(&dev_raa215300_main, RENESAS_PMIC_RAA215300_MAIN_REG_HW_REV,
                         &raa215300_hw_rev, 1);
      if (rc <= 0)
      {
         printf("Failed to read HW rev of Renesas PMIC RAA215300 - retcode %d\r\n", rc);
      }
      else
      {
         printf("Renesas PMIC RAA215300 HW Rev: %02X\n", raa215300_hw_rev);
      }
      i2c_stop(&dev_raa215300_main);
   }

   //=============================================================
   // Renesas PMIC RAA215300 RTC port
   //=============================================================
   //
   // Set the I2C bus filename and slave address
   dev_raa215300_rtc.filename = RIIC3_DEVICE_FILENAME;
   dev_raa215300_rtc.addr = RENESAS_PMIC_RAA215300_RTC_SLAVE_DEV_ADDR;
   // Start the PMIC device on the RTC port.
   rc = i2c_start(&dev_raa215300_rtc);
   if (rc < 0)
   {
      printf("Failed to start Renesas PMIC i2c device RTC port - retcode %d\r\n", rc);
   }
   else
   {
      rc = i2c_readn_reg(&dev_raa215300_rtc, RENESAS_PMIC_RAA215300_RTC_REG_RTC_SR,
                         &raa215300_rtc_sr, 1);
      if (rc <= 0)
      {
         printf("Failed to read register RTC SR of Renesas PMIC RAA215300 - retcode %d\r\n", rc);
      }
      else
      {
         printf("Renesas PMIC RAA215300 RTC SR: %02X\n", raa215300_rtc_sr);
      }
      i2c_stop(&dev_raa215300_rtc);
   }

   //=============================================================
   // TI TLV320AIC23B Audio Codec
   //=============================================================
   //
   // Set the I2C bus filename and slave address
   dev_tlv320aic23b.filename = RIIC3_DEVICE_FILENAME;
   dev_tlv320aic23b.addr = TI_TLV320AIC23B_SLAVE_DEV_ADDR;
   // Start the TI audio codec device
   rc = i2c_start(&dev_tlv320aic23b);
   if (rc < 0)
   {
      printf("Failed to start TI TLV320AIC23B audio codec device - retcode %d\r\n", rc);
   }
   else
   {
      rc = i2c_readn_reg(&dev_tlv320aic23b, TI_TLV320AIC23B_REG_DAI,
                         &tlv320aic23b_dai, 1);
      if (rc <= 0)
      {
         printf("Failed to read DAI register of TI TLV320AIC23B audio codec - retcode %d\r\n", rc);
      }
      else
      {
         printf("TI TLV320AIC23B DAI: %02X\n", tlv320aic23b_dai);
      }
      i2c_stop(&dev_tlv320aic23b);
   }

   //=============================================================
   // Semtech SX150x I2C GPIO port expander for MistySOM
   //=============================================================
   //
   // Set the I2C bus filename and slave address
   dev_sx150x_MistySom.filename = RIIC3_DEVICE_FILENAME;
   dev_sx150x_MistySom.addr = SEMTECH_SX150X_MISTYSOM_SLAVE_DEV_ADDR;
   // Start the Semtech SX150X i2c GPIO port expander device for the MistySOM
   rc = i2c_start(&dev_sx150x_MistySom);
   if (rc < 0)
   {
      printf("Failed to start Semtech SX150X i2c GPIO port expander device for the MistySOM - retcode %d\r\n", rc);
   }
   else
   {
      rc = i2c_readn_reg(&dev_sx150x_MistySom, SEMTECH_SX150X_REG_EV_STS,
                         &sx150x_MistySOM_ev_sts, 1);
      if (rc <= 0)
      {
         printf("Failed to read Semtech SX150X EV_STS register from MistySOM - retcode %d\r\n", rc);
      }
      else
      {
         printf("Semtech SX150X EV_STS register (MistySOM): %02X\n", sx150x_MistySOM_ev_sts);
      }
      i2c_stop(&dev_sx150x_MistySom);
   }

   //=============================================================
   // Semtech SX150x I2C GPIO port expander for MistyCarrier
   //=============================================================
   //
   // Set the I2C bus filename and slave address
   dev_sx150x_MistyCarrier.filename = RIIC3_DEVICE_FILENAME;
   dev_sx150x_MistyCarrier.addr = SEMTECH_SX150X_MISTYCARRIER_SLAVE_DEV_ADDR;
   // Start the Semtech SX150X i2c GPIO port expander device for the MistyCarrier
   rc = i2c_start(&dev_sx150x_MistyCarrier);
   if (rc < 0)
   {
      printf("Failed to start Semtech SX150X i2c GPIO port expander device for the MistyCarrier - retcode %d\r\n", rc);
   }
   else
   {
      rc = i2c_readn_reg(&dev_sx150x_MistyCarrier, SEMTECH_SX150X_REG_EV_STS,
                         &sx150x_MistyCarrier_ev_sts, 1);
      if (rc <= 0)
      {
         printf("Failed to read Semtech SX150X EV_STS register from MistyCarrier - retcode %d\r\n", rc);
      }
      else
      {
         printf("Semtech SX150X EV_STS register (MistyCarrier): %02X\n", sx150x_MistyCarrier_ev_sts);
      }
      i2c_stop(&dev_sx150x_MistyCarrier);
   }

   /*

   //=============================================================
   // Coral MIPI CSI-2 Camera - Omnivision OV5645 video camera sensor
   //=============================================================
   //
   // Set the I2C bus filename and slave address
   dev_ov5645.filename = RIIC0_DEVICE_FILENAME;
   dev_ov5645.addr = OMNIVISION_OV5645_SLAVE_DEV_ADDR;
   // Start the Omnivision ov5645 MIPI CSI-2 video camera device
   rc = i2c_start(&dev_ov5645);
   if (rc < 0)
   {
      printf("Failed to start Omnivision ov5645 MIPI CSI-2 video camera device - retcode %d\r\n", rc);
   }
   else
   {
      rc = i2c_readn_reg(&dev_ov5645, OMNIVISION_OV5645_REG_CHIP_ID_HIGH,
                           &ov5645_chip_id_high, 1);
      if (rc <= 0)
      {
         printf("Failed to read Omnivision ov5645 CHIP_ID HIGH register - retcode %d\r\n", rc);
      }
      else
      {
         rc = i2c_readn_reg(&dev_ov5645, OMNIVISION_OV5645_REG_CHIP_ID_LOW,
                              &ov5645_chip_id_low, 1);
         if (rc <= 0)
         {
            printf("Failed to read Omnivision ov5645 CHIP_ID LOW register - retcode %d\r\n", rc);
         }
         else
         {
            printf("Omnivision ov5645 CHIP ID: %04X\n", (uint16_t)(ov5645_chip_id_high << 8 | ov5645_chip_id_low));
         }
      }
      i2c_stop(&dev_ov5645);
   }


   //=============================================================
   // Renesas HS3001 Humidity and Temp Sensor PMOD (only for Smarc - not MistyCarrier))
   //=============================================================
   //
   // Set the I2C bus filename and slave address
   dev_hs3001.filename = RIIC3_DEVICE_FILENAME;
   dev_hs3001.addr =     RENESAS_HS3001_SLAVE_DEV_ADDR;
   // Start the HS3001 PMOD I2C device (Smarc carrier only)
   rc = i2c_start(&dev_hs3001);
   if (rc < 0)
   {
      printf("Failed to start Renesas HS3001 i2c device - retcode %d\r\n", rc);
      return rc;
   }
   hs3001_enter_nvm_programming_mode(&dev_hs3001);
   uint32_t hs3001_chip_id = hs3001_read_chip_id(&dev_hs3001);
   printf("Renesas HS3001 chip id: %ul\n", hs3001_chip_id);
   hs3001_exit_nvm_programming_mode(&dev_hs3001);
   i2c_stop(&dev_hs3001);
*/

   return 0;
}


