import argparse
import sys
import os
from subprocess import call, check_output


def main():
    action = parse_commandline()
    action()


def parse_commandline():
    parser = argparse.ArgumentParser(
        description='A simple program to compile and run OpenCV programs',
        formatter_class=argparse.RawTextHelpFormatter)

    subparsers = parser.add_subparsers(dest='subcommand')
    add_build_parser(subparsers)

    if len(sys.argv) == 1:
        print_help(parser, bail=True)

    args = parser.parse_args()

    subcommands_actions = {
        'build': build_action
    }

    subcommand_action = subcommands_actions.get(args.subcommand)
    if subcommand_action is not None:
        return lambda: subcommand_action(args)
    else:
        print_help(parser, bail=True)


def build_action(args):
    sources = args.sources

    output = 'result.out'
    if args.output is not None:
        output = args.output
    if len(args.sources) == 1:
        if args.output is None:
            src = args.sources[0]
            output = '{}.out'.format(src[:src.rfind('.')])

    is_release = False
    if args.release:
        is_release = True

    to_execute = args.execute
    arguments = args.arguments
    is_verbose = args.verbose

    cc = ['g++', '-std=c++14']

    flags = [
        '-ggdb',
        '-pipe',
        '-Wundef',
        '-Wstrict-overflow=5',
        '-Wsign-promo',
        '-Woverloaded-virtual',
        '-Wold-style-cast',
        '-Wctor-dtor-privacy',
        '-Wformat=2',
        '-Winvalid-pch',
        '-Wmissing-include-dirs',
        '-Wpacked',
        '-Wpadded',
        '-Wall',
        '-Wextra',
        '-pedantic',
        '-Wdouble-promotion',
        '-Wshadow',
        '-Wfloat-equal',
        '-Wcast-align',
        '-Wcast-qual',
        '-Wwrite-strings',
        '-Wconversion',
        '-Wsign-conversion',
        '-Wmissing-declarations',
        '-Wredundant-decls',
        '-Wdisabled-optimization',
        '-Winline',
        '-Wswitch-default',
        '-Wswitch-enum',
        '-Wuseless-cast',
        '-Wlogical-op',
        '-Wzero-as-null-pointer-constant',
        '-Wnoexcept',
        '-Wstrict-null-sentinel']

    if is_release:
        flags = ['-O2', '-pipe', '-s', '-DNDEBUG', '-Wall',
                 '-D_FORTIFY_SOURCE=1', '-fstack-protector-strong'
                 '-Wdisabled-optimization', '-Wstack-protector', '-Winline']

    opencv_cflags_libs_raw = check_output(
        ['pkg-config', 'opencv', '--cflags', '--libs'])
    opencv_cflags_libs = opencv_cflags_libs_raw.decode().split()

    compiler_call = cc + flags + ['-o', output] + sources + opencv_cflags_libs

    if is_verbose:
        print('Compiler call:')
        print(' '.join(compiler_call), end='\n\n')

    retcode = call(compiler_call)
    if retcode != 0:
        print('Failed building check your code', file=sys.stderr)
        exit(1)

    if to_execute:
        execute_arguments = [os.path.abspath(output)]
        if arguments is not None:
            execute_arguments += arguments

        if is_verbose:
            print('Program call:')
            print(' '.join(execute_arguments))
        call(execute_arguments)


def add_build_parser(subparsers):
    build_parser = subparsers.add_parser(
        'build', description='Use this sub-command to build the OpenCV program')

    build_parser.add_argument(
        '-s',
        '--sources',
        required=True,
        metavar='SOURCE_FILE',
        type=str,
        dest='sources',
        nargs='+',
        help='OpenCV C++ source files')

    build_parser.add_argument(
        '-o',
        '--output',
        required=False,
        metavar='OUTPUT_FILE',
        type=str,
        dest='output',
        help="OpenCV C++ output file")

    build_parser.add_argument(
        '-a',
        '--arguments',
        required=False,
        metavar='ARGUMENT',
        type=str,
        dest='arguments',
        nargs='+',
        help='arguments to pass to the output file')

    exclusive_compilation_mode_group = build_parser.add_mutually_exclusive_group(
        required=False)

    exclusive_compilation_mode_group.add_argument(
        '-r',
        required=False,
        dest='release',
        action='store_true',
        help='Enable release compilation')

    exclusive_compilation_mode_group.add_argument(
        '-d',
        required=False,
        dest='debug',
        action='store_true',
        help='Enable debug compilation')

    build_parser.add_argument(
        '-x',
        required=False,
        dest='execute',
        action='store_true',
        help='Enable automatic execution of the output file')

    build_parser.add_argument(
        '-v',
        required=False,
        dest='verbose',
        action='store_true',
        help='Enable verbose mode')


def print_help(parser, message=None, bail=False):
    if message is not None:
        print('Error Message: {}'.format(message), file=sys.stderr)

    parser.print_help(file=sys.stderr)

    if bail:
        exit(1)


if __name__ == "__main__":
    main()
