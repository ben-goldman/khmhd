# Presentation notes

## Neutron Stars

- Dense remains of collapsed stars (with supernovae)

## Neutron Star Collisions

- Spiral together due to gravitational waves (+LIGO)
- Eventually touch and coalesce
- Emit particle jets and gamma rays

- What happens during coalescence? How do the energy levels necessary to have jets arise?

## Review of Literature

- Attempts to simulate the coalescence phase using computers
- Program in: fluid dynamics, mhd, general relativity
- Simulate entire neutron star collision

- Contributions:
    - B field may be developing in a KHI at interface region
    - B field develops by a dynamo process

- Issues:
    - The entire system is too large to fully simulate
    - Approximating rather than simulating the small-scale fluid dynamics (critical because B develops on small scales)

## Kelvin-Helmholtz Instability

- Cloud demonstration?
- Velocity shear causes swirls to develop
- Unstable: perturbations always grow
- Eventually becomes turbulent

## Dynamos

- Navier-Stokes + Maxwell's equations = Magnetohydrodynamics
- Magnetic fields are organized into field lines
- In MHD, field lines behave like rubber bands, whose tension is analogous to the field strength
- Stretching a field line causes it to strengthen
- In a dynamo, fluid motion allows field lines to be continuously stretched

## Computer Simulations

- 3 parts: initial conditions, physics implementation and parameters, output analysis

- Initial conditions: pac-man screen, shear instability, white noise perturbation
- Physics implementation/parameters: Choice of viscosity, resistivity, density, and resolution. Spectral solver?
- Data analysis: computing energy

## Visual Results

- Describe plot metadata: U and B fields, time-slices, color scales
- Initial conditions: shear flow
- Instability develops similar to clouds picture, magnetic field lines get stretched
- Eventually becomes turbulent with magnetic field continuing to strengthen
- Kinetic energy dies down, magnetic energy stays large

## Dynamo Growth Rate

- Computations: mean energy -> time derivative

## Conclusions

## Next Steps
