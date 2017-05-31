#!/bin/csh -f

# This script was invoked from the faces directory.

foreach f ( [a-z]* )
  echo $f
  cd /afs/cs/project/theo-8/faceimages/faces/$f

  foreach g ( * )
    set base = `echo $g | sed -e "s/.bad//g" -e "s/.pgm//g"`
    set tail = $g:e
    set normal = $base"_1."$tail
    set half = $base"_2."$tail
    set quarter = $base"_4."$tail

    /usr/misc/.pbm/bin/pnmdepth 255 $g > /tmp/foo.pgm
    /usr/misc/.pbm/bin/pnmscale 0.5 /tmp/foo.pgm > /tmp/foo2.pgm
    /usr/misc/.pbm/bin/pnmscale 0.25 /tmp/foo.pgm > /tmp/foo4.pgm

    echo $normal
    /usr/misc/.pbm/bin/pnmnoraw /tmp/foo.pgm > $normal
    echo $half
    /usr/misc/.pbm/bin/pnmnoraw /tmp/foo2.pgm > $half
    echo $quarter
    /usr/misc/.pbm/bin/pnmnoraw /tmp/foo4.pgm > $quarter

    rm -f /tmp/foo*.pgm

  end

end

exit 0
