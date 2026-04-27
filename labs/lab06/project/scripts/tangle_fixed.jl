#!/usr/bin/env julia
# tangle.jl - Generate reports from Literate scripts

using DrWatson
@quickactivate

using Literate

function main()
    if length(ARGS) == 0
        println("""
        Usage: julia tangle.jl <path_to_script>
        
        Examples:
          julia tangle.jl scripts/daisyworld_literary.jl
        """)
        return
    end
    
    script_path = ARGS[1]
    
    if !isfile(script_path)
        error("File not found: $script_path")
    end
    
    script_name = splitext(basename(script_path))[1]
    
    println("🔄 Generating from: $script_path")
    
    # Create output directories
    clean_dir = scriptsdir("clean")
    mkpath(clean_dir)
    
    markdown_dir = projectdir("markdown", script_name)
    mkpath(markdown_dir)
    
    notebooks_dir = projectdir("notebooks", script_name)
    mkpath(notebooks_dir)
    
    # Clean script (without comments) - DON'T execute
    Literate.script(script_path, clean_dir; credit=false)
    println("  ✅ Clean script: $(clean_dir)/$(script_name).jl")
    
    # Quarto document - DON'T execute
    Literate.markdown(script_path, markdown_dir; 
                     flavor=Literate.QuartoFlavor(),
                     name=script_name, 
                     credit=false,
                     execute=false)
    println("  ✅ Quarto: $(markdown_dir)/$(script_name).qmd")
    
    # Jupyter notebook - DON'T execute
    Literate.notebook(script_path, notebooks_dir, 
                     name=script_name; 
                     execute=false, 
                     credit=false)
    println("  ✅ Notebook: $(notebooks_dir)/$(script_name).ipynb")
    
    println("\n✅ Done! All files created.")
end

main()
