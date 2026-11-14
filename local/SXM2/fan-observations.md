# SXM2 adapter fan observations

2026-11-14, reported by the user for the RTE16SXM2B-S adapter:

- The fan header marked **02** is known to react to GPU load: its connected fan
  changes speed as load changes.
- The fan header marked **04** does not appear to react to GPU load. This is an
  observation, not a confirmed diagnosis of the header or fan.
- The user plans to shut down the machine and move one fan to the header marked
  **01** to test its behavior. The result is not yet known.

These are physical fan-header markings, not the numbered LED callouts in the
manual. The manual describes load-based fan speed control but does not document
individual header behavior or a software control interface.