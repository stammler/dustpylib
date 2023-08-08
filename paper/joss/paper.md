---
title: 'DustPyLib: A Library of DustPy Extensions'
tags:
  - Python
  - astronomy
  - planet formation
authors:
  - name: Sebastian M. Stammler^[corresponding author]
    orcid: 0000-0002-1589-1796
    affiliation: 1
  - name: Matías Gárate
    orcid: 0000-0001-6802-834X
    affiliation: 3
  - name: Tilman Birnstiel
    orcid: 0000-0002-1899-8783
    affiliation: 1, 2
affiliations:
  - name: University Observatory, Faculty of Physics, Ludwig-Maximilians-Universität München, Scheinerstr. 1, 81679 Munich, Germany
    index: 1
  - name: Exzellenzcluster ORIGINS, Boltzmannstr. 2, D-85748 Garching, Germany
    index: 2
  - name: Max-Planck-Institut für Astronomie, Königstuhl 17, 69117 Heidelberg, Germany
    index : 3
date: 20 May 2023
bibliography: paper.bib
---

# Summary

`DustPy` is a Python package for dust evolution in protoplanetary disks [@stammler2022b]. It simulates the collisional growth of dust in gas and dust disks around young stellar objects from micrometer-sized particles to meter-sized boulders and eventually to planets in circumstellar gas and dust disks. Results of simulations can then be compared to observations of circumstellar disk to retrieve dust masses and size distributions linking observations to planet formation theories.

# Statement of need

Since `DustPy` is based on the `Simframe` framework for scientific simulations [@stammler2022a], it is rather simple to interchange and modify every aspect of the software. Numerous peer-review publications already made use of the interchangeability of `DustPy` modules investigating a variety of open questions of protoplanetary disk evolution. For example the formation of planetesimals in density enhancements [@stammler2019; @miller2021], the permeability of gaps opened by planets [@stammler2023], the influence of dust particle size distributions on the gas ionization fraction and therefore on the magnetorotational instability [@delage2023], or the effect erosion on dust evolution on young protoplanetary disks [@xu2023].

The purpose of `DustPyLib` is to provide a library of `DustPy` modifications and extensions that users can easily import and use in their own `DustPy` setups or which can help to reproduce previous results. `DustPyLib` is meant to be a community project, where users can contribute their own customizations making them available for others to be used in their own projects. In that sense it is similar to the `REBOUNDx` extension package [@tamayo2020] for the N-body solver `REBOUND` [@rein2012].

# Contribution

To contribute extensions users can open pull requests with their additions on the `DustPyLib` GitHub repository. Contributors are asked to add the scope and a user manual with examples in the form of a Jupyter notebook, which can be added to the documentation. Examples should only demonstrate the setup but not run full simulations, since the documentations is publically hosted by [Read the Docs](https://readthedocs.org/).

If extension already have been published in a scientific publication, contributors are welcome to add references to the documentation, which users of their extensions can cite.

Contributors should also provide unit tests that are testing typical applications of their extensions. It should be aimed for a code coverage of $100\,\%$.

# Examples

As of the time of this publication `DustPyLib` already contains various extensions and modifications that can be readily used and which are listed here.

## Radiative Transfer

`DustPyLib` contains an interface to the Monte Carlo radiative transfer code RADMC-3D [@dullemond2012]. The interface creates three-dimensional axis-symmetric disk models from `DustPy` simulations and produces RADMC-3D input files. Dust opacities are created with `dsharp_opac` [@birnstiel2018b], which can produce two different dust opacities [@ricci2010; @birnstiel2018a]. Other custom opacities in the `RADMC-3D` input format can be produced with `OpTool` [@dominik2021].

![Radiative transfer calculation of the protoplanetary disk of a Solar System analogue containing only the planets Jupiter and Saturn made with `RADMC-3D`. The input files for `RADMC-3D` have been created with `DustPyLib` from a `DustPy` simulation. \label{fig:radmc3d}](radmc3d.png)

## Planetary Gaps

Massive planets are exerting torque on the gas in protoplanetary disks and can open gaps in the gas disk. These gaps can be imposed on `DustPy` simulations by modifying the kinematic viscosity according the desired gap shapes. `DustPyLib` as of now contains two recipes for creating gap shapes by planets [@kanagawa2017; @duffell2020].

## Planetesimal Formation

If dust particles are concentrated above a critical threshold, hydrodynamic instabilities can trigger gravitational collapse of dust clouds leading to the formation of planetesimals. `DustPyLib` so far contains three different flavors of implementing planetesimal formation into `DustPy` simulations [@drazkowska2016; @schoonenberg2018; @miller2021].

## Auxiliary Modules

Certain `DustPy` simulation setups require a radial grid that has been refined at certain locations above the default setup. `DustPyLib` contains a simple method of refining the radial grid with custom refinement factors.

# Acknowledgements

The authors acknowledge funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme under grant agreement No 714769 and funding by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under grants 361140270, 325594231, and Germany's Excellence Strategy - EXC-2094 - 390783311.

# References