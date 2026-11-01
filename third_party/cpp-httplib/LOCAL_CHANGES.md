# Local changes to cpp-httplib 0.54.1

`httplib.h` is upstream 0.54.1 with one change. Re-apply it when upgrading.

- **Keep-alive buffer** (`Server::set_keep_alive_timeout_buffer`, member `keep_alive_timeout_buffer_sec_`): an idle
  connection stays open that many seconds past the `Keep-Alive: timeout=` value the server advertises. The header
  still carries `set_keep_alive_timeout`'s value; the two `process_server_socket` calls (plain and SSL) wait for
  the sum. A client that retires a pooled connection at the advertised deadline then always closes first and
  never writes a request onto a connection the server is closing (nginx's second `keepalive_timeout` parameter
  and Node's `keepAliveTimeoutBuffer` do the same). `infernix-serve` and `infernix-launcher` set 10 s.
