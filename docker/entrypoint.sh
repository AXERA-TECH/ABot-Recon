#!/usr/bin/env bash
# Make the host-mounted Axera runtime visible to the dynamic loader, then start the service
# (or run the given command).
set -e
case "${ABOT_DEVICE:-axcl}" in
  axcl)  LIBDIR=/usr/lib/axcl; PROBE=$LIBDIR/libaxcl_rt.so
         HINT="mount the host AXCL runtime: -v /usr/lib/axcl:/usr/lib/axcl:ro -v /usr/bin/axcl:/usr/bin/axcl:ro --device /dev/axcl_host --device /dev/ax_mmb_dev --device /dev/msg_userdev"
         export PATH=/usr/bin/axcl:$PATH ;;
  ax650) LIBDIR=/soc/lib; PROBE=$LIBDIR/libax_engine.so
         HINT="mount the board runtime: -v /soc:/soc:ro --privileged" ;;
  *)     LIBDIR=""; PROBE=""; HINT="" ;;
esac
if [ -n "$LIBDIR" ] && [ -e "$PROBE" ]; then
  echo "$LIBDIR" > /etc/ld.so.conf.d/axera.conf && ldconfig
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$LIBDIR"
fi
if [ "$#" -gt 0 ]; then exec "$@"; fi
[ -e "$PROBE" ] || { echo "$PROBE not found: $HINT" >&2; exit 2; }
exec bash /app/start_service.sh
