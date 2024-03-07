#!/bin/nu

const examples = 'target/release/examples'
const cli = ($examples | path join cli)
const accuracy = ($examples | path join accuracy)
const k = 256
const trials = 30

const methods = [
    [name, cli_args, dither_args];
    ['Wu - sRGB', [quantette -t 4], [--dither]]
    ['K-means - Oklab', [quantette -t 4 --kmeans --colorspace oklab], [--dither]]
    [imagequant, [imagequant -t 4 -q 100], [--dither-level 1.0]]
    [color_quant, [neuquant --sample-frac 10], null]
    [exoquant, [exoquant], [--dither]]
]

const images = [
    Akihabara.jpg
    Boothbay.jpg
    Hokkaido.jpg
    'Jewel Changi.jpg'
    Louvre.jpg
]

def main [--dither, --no-dither] {
    let dithers = match [$dither $no_dither] {
        [true true] => { error make { msg: 'the --dither and --no-dither flags are exclusive' } }
        [true false] => [true]
        [false true] => [false]
        [false false] => [false true]
    }

    cargo b -r --example cli o+e> /dev/null
    cargo b -r --example accuracy o+e> /dev/null

    let output = mktemp -t quantette_benchmark_output.XXX --suffix .png

    let table = $images | wrap Image | insert path {|img| [img unsplash img Original $img.Image] | path join }

    for dither in $dithers {
        print (if $dither { '# With Dithering' } else { '# Without Dithering' })
        print ''

        print '## Time'
        print ''

        let methods = if $dither {
            $methods | where dither_args != null
        } else {
            $methods | update dither_args []
        }

        $methods
        | reduce -f $table {|method, table|
            $table | insert $method.name {|image|
                1..$trials
                | each {
                    ^$cli $image.path -o $output --verbose -k $k ...$method.cli_args ...$method.dither_args
                    | lines
                    | parse 'quantization and remapping took {time}ms'
                    | get time.0
                    | into int
                }
                | math avg
                | math round
                | into int
            }
        }
        | reject path
        | to md --pretty
        | print

        print ''

        print '## Accuracy/DSSIM'
        print ''

        $methods
        | reduce -f $table {|method, table|
            $table | insert $method.name {|image|
                ^$cli $image.path -o $output --verbose -k $k ...$method.cli_args ...$method.dither_args
                ^$accuracy compare $image.path $output | into float | math round -p 6
            }
        }
        | reject path
        | to md --pretty
        | print

        print ''
    }

    rm $output
}
