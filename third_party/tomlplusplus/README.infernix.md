# Vendored toml++

Infernix vendors the upstream toml++ single-header distribution, the same copy NInfer vendors:

- upstream: <https://github.com/marzer/tomlplusplus>
- tag: `v3.4.0`
- commit: `30172438cee64926dc41fdd9c11fb3ba5b2ba9de`
- license: MIT; see [LICENSE](LICENSE)

`toml.hpp` and `LICENSE` are unchanged upstream files. The header supports TOML 1.0 with the
upstream header-only defaults and no further dependencies.

`cmake/Dependencies.cmake` provides `infernix::toml`. Only the configuration-file reader
(`src/product/config/config_file.cpp`) links it, privately; TOML types stay out of every other
interface. The build downloads nothing and does not look for a system TOML package.
