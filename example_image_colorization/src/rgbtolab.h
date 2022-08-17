/*
 * convert between pixel formats: RGB <-> XYZ <-> LAB
 * adapted from https://stackoverflow.com/a/38393116
 */
#pragma once

#include <stdio.h>
#include <math.h>

/// convert RGB 0-255 to XYZ 0-1
void rgbtoxyz(int inR, int inG, int inB,
              float *outX, float *outY, float *outZ) {
	float var_R = (inR / 255.0f);
    float var_G = (inG / 255.0f);
    float var_B = (inB / 255.0f);

	if(var_R > 0.04045f) {
        var_R = powf(((var_R + 0.055f) / 1.055f), 2.4f);
	}
	else {
        var_R = var_R / 12.92f;
	}

	if(var_G > 0.04045) {
        var_G = powf(((var_G + 0.055f) / 1.055f), 2.4f);
	}
	else {
        var_G = var_G / 12.92f;
	}

	if(var_B > 0.04045f) {
        var_B = powf(((var_B + 0.055f) / 1.055f), 2.4f);
	}
	else {
        var_B = var_B / 12.92f;
	}

    var_R = var_R * 100;
    var_G = var_G * 100;
    var_B = var_B * 100;

    // Observer = 2°, Illuminant = D65
    *outX = var_R * 0.4124f + var_G * 0.3576f + var_B * 0.1805f;
    *outY = var_R * 0.2126f + var_G * 0.7152f + var_B * 0.0722f;
    *outZ = var_R * 0.0193f + var_G * 0.1192f + var_B * 0.9505f;
}

/// convert XYZ 0-1 to LAB 0-1
void xyztolab(float inX, float inY, float inZ,
              float *outL, float *outA, float *outB) {
    float var_X = inX / 95.047;
    float var_Y = inY / 100.0;
    float var_Z = inZ / 108.883;

	if(var_X > 0.008856) {
        var_X = powf(var_X, (1.0f / 3));
	}
	else {
        var_X = (7.787 * var_X) + (16.0f / 116);
	}

	if(var_Y > 0.008856) {
        var_Y = powf(var_Y, (1.0f / 3));
	}
	else {
        var_Y = (7.787 * var_Y) + (16.0f / 116);
	}

	if(var_Z > 0.008856) {
        var_Z = powf(var_Z, (1.0f / 3));
	}
	else {
        var_Z = (7.787 * var_Z) + (16.0f / 116);
	}

    *outL = (116 * var_Y) - 16;
    *outA = 500 * (var_X - var_Y);
    *outB = 200 * (var_Y - var_Z);
}

/// convert RGB 0-255 to LAB 0-1
void rgbtolab(int inR, int inG, int inB,
              float *outL, float *outA, float *outB) {
	float X, Y, Z;
	rgbtoxyz(inR, inG, inB, &X, &Y, &Z);
	xyztolab(X, Y, Z, outL, outA, outB);
}

/// convert LAB 0-1 to XYZ 0-1
void labtoxyz(float inL, float inA, float inB,
              float *outX, float *outY, float *outZ) {
    float var_Y = (inL + 16) / 116;
    float var_X = (inA / 500) + var_Y;
    float var_Z = var_Y - (inB / 200);

	if(powf(var_Y, 3.f) > 0.008856) {
        var_Y = powf(var_Y, 3.f);
	}
	else {
        var_Y = (var_Y - (16 / 116)) / 7.787;
	}

	if(powf(var_X, 3.f) > 0.008856) {
        var_X = powf(var_X, 3.f);
	}
	else {
        var_X = (var_X - (16 / 116)) / 7.787;
	}

	if(powf(var_Z, 3.f) > 0.008856) {
        var_Z = powf(var_Z, 3.f);
	}
	else {
        var_Z = (var_Z - (16/116)) / 7.787;
	}

	// Observer = 2°, Illuminant = D65
    *outX = var_X * 95.047;
    *outY = var_Y * 100.0;
    *outZ = var_Z * 108.883;
}

/// convert XYZ 0-1 to RGB 0-255
void xyztorgb(float inX, float inY, float inZ,
              int *outR, int *outG, int *outB) {
    float var_X = inX / 100;
    float var_Y = inY / 100;
    float var_Z = inZ / 100;

    float var_R = var_X *  3.2406 + (var_Y * -1.5372) + var_Z * (-0.4986);
    float var_G = var_X * (-0.9689) + var_Y *  1.8758 + var_Z *  0.0415;
    float var_B = var_X *  0.0557 + var_Y * (-0.2040) + var_Z *  1.0570;

	if(var_R > 0.0031308) {
        var_R = 1.055 * powf(var_R, (1.0f / 2.4))  - 0.055;
	}
	else {
        var_R = 12.92 * var_R;
	}

	if(var_G > 0.0031308) {
        var_G = 1.055 * powf(var_G, (1.0f / 2.4)) - 0.055;
	}
	else {
        var_G = 12.92 * var_G;
	}

	if(var_B > 0.0031308) {
        var_B = 1.055 * powf(var_B, (1.0f / 2.4)) - 0.055;
	}
	else {
        var_B = 12.92 * var_B;
	}

    *outR = (int)(var_R * 255);
    *outG = (int)(var_G * 255);
    *outB = (int)(var_B * 255);
}

/// convert LAB 1-0 to RGB 0-255
void labtorgb(float inL, float inA, float inB,
              int *outR, int *outG, int *outB) {
	float X, Y, Z;
	labtoxyz(inL, inA, inB, &X, &Y, &Z);
	xyztorgb(X, Y, Z, outR, outG, outB);
}
