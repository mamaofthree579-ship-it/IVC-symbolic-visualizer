import numpy as np

def fresnel_propagation(height, wavelength=532e-9, dx=10e-6, z=0.15):
    field = np.exp(1j * 2*np.pi * height)
    ny, nx = field.shape

    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    FX, FY = np.meshgrid(np.fft.fftshift(fx), np.fft.fftshift(fy))

    Hf = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))

    U2 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F * Hf)))

    return np.abs(U2)**2
