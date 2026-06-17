$pdf_mode = 1;
$out_dir = 'build';
$aux_dir = 'build';

# Ensure FrontMatter and Chapters subdirs exist in build before each run
$pre_tex_code = '';
system("mkdir -p build/FrontMatter build/Chapters 'build/Back Matter'");
