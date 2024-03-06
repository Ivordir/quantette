#!/bin/nu

const image = 'img/CQ100/img/calaveras.png'
const examples = 'target/release/examples'
const cli = ($examples | path join cli)
const accuracy = ($examples | path join accuracy)

const methods = [
    [name, file, cli_args];
    [Gimp, gimp, null]
    ['Wu - sRGB', wu_srgb, []]
    ['K-means - Oklab', kmeans_oklab, [--kmeans --colorspace oklab]]
]

def main [-k: int, --dither, --no-dither] {
    let dithers = match [$dither $no_dither] {
        [true true] => { error make { msg: 'the --dither and --no-dither flags are exclusive' } }
        [true false] => [true]
        [false true] => [false]
        [false false] => [false true]
    }

    let ks = if $k == null {
        [16 64 256]
    } else {
        [$k]
    }

    cargo b -r --example cli o+e> /dev/null
    cargo b -r --example accuracy o+e> /dev/null

    for dither in $dithers {
        print (if $dither { '# With Dithering' } else { '# Without Dithering' })
        print ''

        let dither_suffix = if $dither { '_dither' } else { '' }
        let dither_arg = if $dither { [ '--dither' ] } else { [] }

        for k in $ks {
            print $'## ($k) Colors'
            print ''

            $methods
            | each {|method|
                let result = $'img/($method.file)_($k)($dither_suffix).png'
                let output = 'docs' | path join $result
                if $method.cli_args != null {
                    ^$cli $image -o $output -k $k quantette ...$dither_arg ...$method.cli_args
                }
                let dssim = ^$accuracy compare $image $output | into float | math round -p 6
                {
                    Method: $method.name
                    DSSIM: $dssim
                    Result: $"![]\(($result)\)"
                }
            }
            | to md --pretty
            | print

            print ''
        }
    }
}
