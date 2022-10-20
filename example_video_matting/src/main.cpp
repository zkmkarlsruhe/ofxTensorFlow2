/*
 * Example made with love by Natxopedreira 2021
 * https://github.com/natxopedreira
 * Updated by members of the ZKM | Hertz-Lab 2021
 */

#include "ofMain.h"
#include "ofApp.h"

//========================================================================
int main() {
	ofGLWindowSettings settings;
	settings.setGLVersion(3, 2); // try also 2, 1
	settings.setSize(1280, 720);
	settings.windowMode = OF_WINDOW;
	ofCreateWindow(settings);
	ofRunApp(new ofApp());
}
