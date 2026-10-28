from typing import IO
from os import path


from models.models import GenLibParseResult, SizeStandardParseResult


def parse_file(file: IO[bytes], filename: str) -> tuple[list[SizeStandardParseResult], list[GenLibParseResult]]:
    ext = path.splitext(filename)[1].lower()

    if ext == '.frf':
        from lib.parsing.parsing_frf import parse_file
        return parse_file(file, filename)

    # if ext == '.fsa':
    #     from lib.parsing.parsing_fsa import parse_file
    #     return parse_file(file, filename)

    raise ValueError(f"Неподдерживаемый формат файла: {ext}")
