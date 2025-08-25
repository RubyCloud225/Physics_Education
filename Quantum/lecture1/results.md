## In the Double-Slit Setup
	•	Both slits open (no measurement):
Each slit contributes a complex amplitude:
$${\Psi(x) = \Psi_1(x) + \Psi_2(x)}$$

### → interference pattern forms from cross-terms.
	•	Measurement (which slit?):
If you “peek” to see which slit the particle goes through, the cross-terms vanish:
$${P(x) = |\Psi_1(x)|^2 + |\Psi_2(x)|^2}$$

### → looks like two blurred single-slit patterns added together (no fringes).
	•	Blocking one slit (or one amplitude = 0):
Then the wavefunction is just:
$${\Psi(x) = \Psi_1(x) \quad \text{or} \quad \Psi_2(x)}$$
### single-slit diffraction pattern only, no interference.


## Why It Happens in Simulation

If your code gives you one slit contributing nothing while the other contributes a distribution:
	•	You may have accidentally set the amplitude of one slit to zero (e.g., by applying measure=True incorrectly or normalizing oddly).
	•	In real terms, that means you’ve “blocked” or “collapsed” one path, so there’s no way for interference to occur.

### Set up is zero for slit 1

essentially blocking off one of the channels.

amp_slit1 = np.exp(1j * 0)
amp_slit2 = np.exp(1j * phase_diff)

Both are nonzero by default. If you set one of them to 0 (or normalize incorrectly), then you’re simulating blocking that slit.