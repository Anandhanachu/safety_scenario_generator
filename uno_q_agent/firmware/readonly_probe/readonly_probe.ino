// Optional read-only probe. Upload explicitly using Arduino App Lab/IDE.
// Hardware compilation and behavior require verification on the actual UNO Q.
#include <Arduino_RouterBridge.h>

String capabilities() { return "safescenario-readonly-v1"; }
String ping() { return "pong"; }

void setup() {
  Bridge.begin();
  Bridge.provide_safe("safescenario_capabilities", capabilities);
  Bridge.provide_safe("safescenario_ping", ping);
}

void loop() {}
