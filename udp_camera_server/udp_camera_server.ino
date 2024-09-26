#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_camera.h"

const char* ssid = "xxxxxxx";
const char* password = "xxxxxx";
const char* udpAddress = "192.168.x.x";  // UDP istemcisinin IP adresi
const int udpPort = 12345;  // UDP. portu

WiFiUDP udp;

void cameraInit(void);

void setup() {
  Serial.begin(115200);

  cameraInit();  // Kamera konfigürasyonu yapıldı

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("Connected to WiFi");
  Serial.println(WiFi.localIP());

  // UDP'yi başlat
  udp.begin(udpPort);

  // UDP verilerini göndermek için FreeRTOS görevi oluşturun
  xTaskCreate(sendUdpDataTask, "SendUdpData", 4096, NULL, 1, NULL);
}

void loop() {
  // Main loop can be used for other tasks
  delay(1000);
}

void sendUdpDataTask(void* parameter) {
    const int maxPacketSize = 1400;  // UDP paketi başına maksimum boyut
    while (true) {
        camera_fb_t* fb = esp_camera_fb_get();  // Çerçeve verisini al

        if (!fb) {
            Serial.println("Kamera verisi alınamadı");
            vTaskDelay(1000 / portTICK_PERIOD_MS);
            continue;
        }

        uint32_t totalBytes = fb->len;
        uint32_t bytesSent = 0;

        while (bytesSent < totalBytes) {
            uint32_t bytesToSend = min((uint32_t)maxPacketSize, totalBytes - bytesSent);
            udp.beginPacket(udpAddress, udpPort);
            udp.write(fb->buf + bytesSent, bytesToSend);
            udp.endPacket();

            bytesSent += bytesToSend;
        }

        Serial.printf("Sent %d bytes in multiple packets\n", totalBytes);

        esp_camera_fb_return(fb);  // Çerçeve belleğini serbest bırak
        vTaskDelay(33 / portTICK_PERIOD_MS);  // 30 fps için 33ms bekle
    }
}


void cameraInit(void) {
  // Kamera ayarları (kodunuzdaki gibi)
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = CAMD2;
  config.pin_d1 = CAMD3;
  config.pin_d2 = CAMD4;
  config.pin_d3 = CAMD5;
  config.pin_d4 = CAMD6;
  config.pin_d5 = CAMD7;
  config.pin_d6 = CAMD8;
  config.pin_d7 = CAMD9;
  config.pin_xclk = CAMXC;
  config.pin_pclk = CAMPC;
  config.pin_vsync = CAMV;
  config.pin_href = CAMH;
  config.pin_sscb_sda = CAMSD;
  config.pin_sscb_scl = CAMSC;
  config.pin_pwdn = -1;
  config.pin_reset = -1;

  config.xclk_freq_hz = 15000000;
  config.frame_size = FRAMESIZE_UXGA;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.fb_count = 2;
  config.jpeg_quality = 30;

  if (!psramFound()) {
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Kamera başlatma hatası 0x%x", err);
    return;
  }

  sensor_t* s = esp_camera_sensor_get();
}
