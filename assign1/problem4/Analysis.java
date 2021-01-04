// Assignment 1 (CS610)
// Name: Ayush Kumar
// Roll No: 170195

// java org.antlr.v4.gui.TestRig LoopNest tests -gui < UnitTestCases.java

import java.util.*;
import static java.util.stream.Collectors.*;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.lang.Math;

import org.antlr.v4.runtime.tree.ParseTreeProperty;
import org.antlr.v4.runtime.tree.TerminalNode;

import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.Vocabulary;
// import org.antlr.v4.runtime.tree.*;
import org.antlr.v4.runtime.Token;

// FIXME: You should limit your implementation to this class. You are free to add new auxilliary classes. You do not need to touch the LoopNext.g4 file.
class Analysis extends LoopNestBaseListener {

    // Possible types
    enum Types {
        Byte, Short, Int, Long, Char, Float, Double, Boolean, String
    }

    // Type of variable declaration
    enum VariableType {
        Primitive, Array, Literal
    }

    // Types of caches supported
    enum CacheTypes {
        DirectMapped, SetAssociative, FullyAssociative,
    }

    // auxilliary data-structure for converting strings
    // to types, ignoring strings because string is not a
    // valid type for loop bounds
    final Map<String, Types> stringToType = Collections.unmodifiableMap(new HashMap<String, Types>() {
        private static final long serialVersionUID = 1L;

        {
            put("byte", Types.Byte);
            put("short", Types.Short);
            put("int", Types.Int);
            put("long", Types.Long);
            put("char", Types.Char);
            put("float", Types.Float);
            put("double", Types.Double);
            put("boolean", Types.Boolean);
            put("String", Types.String);
        }
    });

    // auxilliary data-structure for mapping types to their byte-size
    // size x means the actual size is 2^x bytes, again ignoring strings
    final Map<Types, Integer> typeToSize = Collections.unmodifiableMap(new HashMap<Types, Integer>() {
        private static final long serialVersionUID = 1L;

        {
            put(Types.Byte, 0);
            put(Types.Short, 1);
            put(Types.Int, 2);
            put(Types.Long, 3);
            put(Types.Char, 1);
            put(Types.Float, 2);
            put(Types.Double, 3);
            put(Types.Boolean, 0);
            put(Types.String, 1);
        }
    });

    // Map of cache type string to value of CacheTypes
    final Map<String, CacheTypes> stringToCacheType = Collections.unmodifiableMap(new HashMap<String, CacheTypes>() {
        private static final long serialVersionUID = 1L;

        {
            put("FullyAssociative", CacheTypes.FullyAssociative);
            put("SetAssociative", CacheTypes.SetAssociative);
            put("DirectMapped", CacheTypes.DirectMapped);
        }
    });

    public Analysis() {
        cachemisses_aggr = new ArrayList<HashMap<String, Long>>();
        cachemisses = new HashMap<String, Long>();
        symboltable = new HashMap<String, Object>();
        symboltypes = new HashMap<String, Types>();
        symboldims = new HashMap<String, ArrayList<Long>>();
        loopvars = new ArrayList<String>();
        endvalues = new ArrayList<Integer>();
        strides = new ArrayList<Integer>();
    }

    List<HashMap<String, Long>> cachemisses_aggr;
    HashMap<String, Long> cachemisses;
    HashMap<String, Object> symboltable;
    HashMap<String, Types> symboltypes;
    HashMap<String, ArrayList<Long>> symboldims;
    CacheTypes cachetype;
    int cachepower, blockpower;
    long associativity;
    ArrayList<String> loopvars;
    ArrayList<Integer> endvalues;
    ArrayList<Integer> strides;

    // FIXME: Feel free to override additional methods from
    // LoopNestBaseListener.java based on your needs.
    // Method entry callback
    @Override
    public void enterMethodDeclaration(LoopNestParser.MethodDeclarationContext ctx) {
        // System.out.println("enterMethodDeclaration");
        // A new testcase encountered
        symboltable.clear(); symboltypes.clear(); symboldims.clear();
        cachemisses.clear();
        loopvars.clear(); endvalues.clear(); strides.clear();
    }

    @Override
    public void exitMethodDeclaration(LoopNestParser.MethodDeclarationContext ctx) {
        HashMap<String, Long> cachemisses_copy = new HashMap<String, Long>(cachemisses);
        cachemisses_aggr.add(cachemisses_copy);
        System.out.println();
    }

    @Override
    public void enterMethodDeclarator(LoopNestParser.MethodDeclaratorContext ctx) {
        System.out.printf("##########Testcase: %s##########\n", ctx.children.get(0).getText());
    }

    // End of testcase
    @Override
    public void exitMethodDeclarator(LoopNestParser.MethodDeclaratorContext ctx) {
        // System.out.println("exitMethodDeclarator");
    }

    @Override
    public void exitTests(LoopNestParser.TestsContext ctx) {
        // for(int i = 0; i < cachemisses_aggr.size(); i++) {
        //     for(String key: cachemisses_aggr.get(i).keySet()) {
        //         System.out.printf("%s, %d\n", key, cachemisses_aggr.get(i).get(key));
        //     }
        //     System.out.println();
        // }
        try {
            FileOutputStream fos = new FileOutputStream("Results.obj");
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            // FIXME: Serialize your data to a file
            oos.writeObject(cachemisses_aggr);
            oos.close();
        } catch (Exception e) {
            throw new RuntimeException(e.getMessage());
        }
    }

    @Override
    public void exitLocalVariableDeclaration(LoopNestParser.LocalVariableDeclarationContext ctx) {
        ParseTree variable_declarator;
        variable_declarator = ctx.children.get(1);
        String varname = variable_declarator.getChild(0).getText();
        String basetype = "int"; 
        ArrayList<Long> dims = new ArrayList<Long>(); 
        Object literal_val = 1;
        if (ctx.children.get(0) instanceof LoopNestParser.UnannStringTypeContext) {
            basetype = "String";
        } else if (ctx.children.get(0) instanceof LoopNestParser.UnannTypeContext) {
            ParseTree unann_type = ctx.children.get(0).getChild(0);
            if (unann_type instanceof LoopNestParser.UnannPrimitiveTypeContext) {
                basetype = unann_type.getText();
            } else if (unann_type instanceof LoopNestParser.UnannArrayTypeContext) {
                // ArrayType
                basetype = unann_type.getChild(0).getText(); // unannPrimitiveType
                // dims = unann_type.getChild(0).getText(); // dims
            }
        }
        
        if (variable_declarator.getChild(2) instanceof LoopNestParser.LiteralContext) {
            // integer or string
            ParseTree literal = variable_declarator.getChild(2);
            Token literal_token = ((TerminalNode)literal.getChild(0)).getSymbol();
            String literal_type = LoopNestLexer.VOCABULARY.getSymbolicName(literal_token.getType());
            if (literal_type == "StringLiteral") { // string
                literal_val = literal.getText();
                literal_val = ((String)literal_val).substring(1, ((String)literal_val).length()-1);
            } else if (literal_type == "IntegerLiteral") { // integer
                literal_val = Integer.parseInt(literal.getText());
            }
        } else if (variable_declarator.getChild(2) instanceof LoopNestParser.ArrayCreationExpressionContext) {
            // array expression
            ParseTree dim_exprs = variable_declarator.getChild(2).getChild(2);
            for(int i = 0; i < dim_exprs.getChildCount(); i++) {
                ParseTree dim_size = dim_exprs.getChild(i).getChild(1);
                if (dim_size instanceof LoopNestParser.ExpressionNameContext) {
                    dims.add((long)((Integer)symboltable.getOrDefault(dim_size.getText(), 1)).intValue());
                } else { // IntegerLiteral
                    dims.add(Long.parseLong(dim_size.getText()));
                }
            }
        }

        symboltypes.put(varname, stringToType.get(basetype));
        symboltable.put(varname, literal_val);
        symboldims.put(varname, dims);

        if (varname.equals("cachePower")) cachepower = (int)(symboltable.getOrDefault("cachePower", 0));
        else if (varname.equals("blockPower")) blockpower = (int)(symboltable.getOrDefault("blockPower", 0));
        else if (varname.equals("cacheType")) cachetype = (CacheTypes)(stringToCacheType.get(symboltable.getOrDefault("cacheType", "DirectMapped")));
        else if (varname.equals("setSize")) associativity = (int)(symboltable.getOrDefault("setSize", 1));
    }

    @Override
    public void exitVariableDeclarator(LoopNestParser.VariableDeclaratorContext ctx) {
    }

    @Override
    public void exitArrayCreationExpression(LoopNestParser.ArrayCreationExpressionContext ctx) {
    }

    @Override
    public void exitDimExprs(LoopNestParser.DimExprsContext ctx) {
    }

    @Override
    public void exitDimExpr(LoopNestParser.DimExprContext ctx) {
    }

    @Override
    public void exitLiteral(LoopNestParser.LiteralContext ctx) {
    }

    @Override
    public void exitVariableDeclaratorId(LoopNestParser.VariableDeclaratorIdContext ctx) {
    }

    @Override
    public void exitUnannArrayType(LoopNestParser.UnannArrayTypeContext ctx) {
    }

    @Override
    public void enterDims(LoopNestParser.DimsContext ctx) {
    }

    @Override
    public void exitUnannPrimitiveType(LoopNestParser.UnannPrimitiveTypeContext ctx) {
    }

    @Override
    public void exitNumericType(LoopNestParser.NumericTypeContext ctx) {
    }

    @Override
    public void exitIntegralType(LoopNestParser.IntegralTypeContext ctx) {
    }

    @Override
    public void exitFloatingPointType(LoopNestParser.FloatingPointTypeContext ctx) {
    }

    @Override
    public void exitForStatement(LoopNestParser.ForStatementContext ctx) {
        loopvars.remove(loopvars.size() - 1);
        endvalues.remove(endvalues.size() - 1);
        strides.remove(strides.size() - 1);
    }

    @Override
    public void exitForInit(LoopNestParser.ForInitContext ctx) {
        String loopvar = ctx.children.get(0).getChild(1).getChild(0).getText();
        loopvars.add(loopvar);
    }

    @Override
    public void exitForCondition(LoopNestParser.ForConditionContext ctx) {
        ParseTree relational_expression = ctx.children.get(0);
        if (relational_expression.getChildCount() == 3) {
            String l = relational_expression.getChild(0).getText();
            int r = 0;
            if (relational_expression.getChild(2) instanceof LoopNestParser.ExpressionNameContext) {
                r = (int)symboltable.getOrDefault(relational_expression.getChild(2).getText(), 1);
            } else {
                r = Integer.parseInt(relational_expression.getChild(2).getText());
            }
            endvalues.add(r);
        }
    }

    @Override
    public void exitRelationalExpression(LoopNestParser.RelationalExpressionContext ctx) {
    }

    @Override
    public void exitForUpdate(LoopNestParser.ForUpdateContext ctx) {
        ParseTree simplified_assignment = ctx.children.get(0);
        String l = simplified_assignment.getChild(0).getText();
        int r = 0;
        if (simplified_assignment.getChild(2) instanceof LoopNestParser.ExpressionNameContext) {
            r = ((Integer)symboltable.getOrDefault(simplified_assignment.getChild(2).getText(), 1)).intValue();
        } else {
            r = Integer.parseInt(simplified_assignment.getChild(2).getText());
        }
        strides.add(r);
        // l should be the loop variable
    }

    @Override
    public void exitSimplifiedAssignment(LoopNestParser.SimplifiedAssignmentContext ctx) {
    }

    public void processArrayAccess(String varname, HashMap<String, Integer> indices_to_dimindex) {
        // calculate the number of misses based for this array access and store it in cachemisses[varname]
        // all the following values are in powers of 2
        int bytes_in_word_power = typeToSize.get(symboltypes.get(varname));
        long words_in_cache = 1L << (cachepower - bytes_in_word_power);
        long words_in_block = 1L << (blockpower - bytes_in_word_power);
        long blocks_in_cache = 1L << (cachepower - blockpower);
        long blocks_in_set = 1L << 0;
        if (cachetype == CacheTypes.DirectMapped) blocks_in_set = associativity = 1L << 0;
        else if (cachetype == CacheTypes.FullyAssociative) blocks_in_set = associativity = blocks_in_cache;
        else if (cachetype == CacheTypes.SetAssociative) blocks_in_set = associativity;
        long sets_in_cache = blocks_in_cache / blocks_in_set;
        if (blockpower < bytes_in_word_power) {
            // fetching a word from cache will require fetching multiple blocks
            long blocks_in_word = 1L << (bytes_in_word_power - blockpower);
            if (sets_in_cache >= blocks_in_word) {
                sets_in_cache /= blocks_in_word;
            } else {
                blocks_in_set /= blocks_in_word;
                associativity /= blocks_in_word;
            }
            words_in_block = 1;
            blocks_in_cache = sets_in_cache * blocks_in_set;
        }
        
        
        ArrayList<Long> dims = symboldims.get(varname);

        int loop_idx = loopvars.size() - 1;
        long misses = 0;
        boolean overwrite = false;
        HashMap<Integer, Long> set_order_offsets = new HashMap<Integer, Long>();
        long set_access_stride, used_sets_in_cache, available_sets_in_cache;
        while (loop_idx >= 0 && !indices_to_dimindex.containsKey(loopvars.get(loop_idx))) loop_idx--;
        int inner_dim_idx = indices_to_dimindex.get(loopvars.get(loop_idx));
        // indices contains loopvars[idx]
        long effective_stride = strides.get(loop_idx);
        for (int i = indices_to_dimindex.get(loopvars.get(loop_idx)) + 1; i < dims.size(); i++) {
            effective_stride *= dims.get(i);
        }
        // in the loop at index loop_idx, every (effective_stride)'th word is accessed
        
        // System.out.printf("Words in block: %d\n", words_in_block);
        // System.out.printf("Sets in cache: %d\n", sets_in_cache);
        // System.out.printf("Associativity: %d\n", associativity);
        // System.out.printf("Effective_stride: %d\n", effective_stride);
        if (words_in_block <= effective_stride) {
            // all accesses are misses
            misses = endvalues.get(loop_idx) < strides.get(loop_idx) ? 1 : 
                    endvalues.get(loop_idx) / strides.get(loop_idx);
            set_access_stride = effective_stride / words_in_block;
            // every (effective_stride / words_in_block)'th block is fetched
            if (sets_in_cache < (effective_stride/words_in_block)) {
                used_sets_in_cache = 1;
            } else {
                used_sets_in_cache = Math.min(sets_in_cache / (effective_stride/words_in_block), misses);
            }

            if (sets_in_cache < (effective_stride/words_in_block)) {
                if (associativity < misses) overwrite = true;
            } else if ((sets_in_cache / (effective_stride/words_in_block)) * associativity < misses) {
                overwrite = true;
            }
        } else {
            // every words_in_block / (effective_stride) is a miss
            misses = (endvalues.get(loop_idx)/strides.get(loop_idx)) < (words_in_block/effective_stride) ? 1 :
                    (endvalues.get(loop_idx)/strides.get(loop_idx)) / (words_in_block/effective_stride);
            set_access_stride = 1; // consecutive blocks are fetched, hence consecutive sets will be filled
            used_sets_in_cache = Math.min(sets_in_cache, misses);
            if (sets_in_cache * associativity < misses) overwrite = true;
        }
        loop_idx--;
        available_sets_in_cache = sets_in_cache / set_access_stride;
        while (loop_idx >= 0) {
            // System.out.println(misses);
            // System.out.println(overwrite);
            // System.out.println(loopvars.get(loop_idx));
            if (!indices_to_dimindex.containsKey(loopvars.get(loop_idx))) {
                // this is just a free loop, loop variable not used to index the array
                if (overwrite) misses *= endvalues.get(loop_idx) < strides.get(loop_idx) ? 1 : 
                        (endvalues.get(loop_idx)/strides.get(loop_idx));
            } else {
                // this loop variable is used to index into the array
                // calculate offset
                int curr_dim_idx = indices_to_dimindex.get(loopvars.get(loop_idx));
                long offset = 1;
                effective_stride = strides.get(loop_idx);
                for(int i = curr_dim_idx + 1; i < dims.size(); i++) {
                    offset *= dims.get(i);
                }
                effective_stride *= offset;
                if (words_in_block <= effective_stride || overwrite) {
                    misses *= endvalues.get(loop_idx) < strides.get(loop_idx) ? 1 : 
                        (endvalues.get(loop_idx)/strides.get(loop_idx));
                } else {
                    misses *= (endvalues.get(loop_idx)/strides.get(loop_idx)) < (words_in_block/effective_stride) ? 1 : 
                        (endvalues.get(loop_idx)/strides.get(loop_idx)) / (words_in_block/effective_stride);
                }

                if (indices_to_dimindex.get(loopvars.get(loop_idx)) > inner_dim_idx) {
                    if (words_in_block <= effective_stride) {
                        used_sets_in_cache *= endvalues.get(loop_idx) < strides.get(loop_idx) ? 1 : 
                            (endvalues.get(loop_idx)/strides.get(loop_idx));
                        available_sets_in_cache *= endvalues.get(loop_idx) < strides.get(loop_idx) ? 1 : 
                            (endvalues.get(loop_idx)/strides.get(loop_idx));
                    } else {
                        used_sets_in_cache *= (endvalues.get(loop_idx)/strides.get(loop_idx)) < (words_in_block/effective_stride) ? 1 : 
                            (endvalues.get(loop_idx)/strides.get(loop_idx)) / (words_in_block/effective_stride);
                        available_sets_in_cache *= (endvalues.get(loop_idx)/strides.get(loop_idx)) < (words_in_block/effective_stride) ? 1 : 
                            (endvalues.get(loop_idx)/strides.get(loop_idx)) / (words_in_block/effective_stride);
                    }
                }

                if (indices_to_dimindex.get(loopvars.get(loop_idx)) < inner_dim_idx) {
                    // this loop variable is used to index a lower dimension than the innermost one
                    // System.out.println(overwrite);
                    // System.out.println(used_sets_in_cache);
                    if ((effective_stride/words_in_block) < sets_in_cache && used_sets_in_cache < available_sets_in_cache) {
                        // expand the number of used sets if possible
                        used_sets_in_cache *= endvalues.get(loop_idx) < strides.get(loop_idx) ? 1 : 
                                            (endvalues.get(loop_idx)/strides.get(loop_idx));
                        used_sets_in_cache = Math.min(used_sets_in_cache, available_sets_in_cache);
                    }
                    if (!overwrite && misses > used_sets_in_cache * associativity) overwrite = true;
                    // System.out.println(overwrite);
                } else {
                    // this loop variable is used to index a higher dimension than previous one
                    // we don't care about the overwrite flag here                    
                }
            }
            loop_idx--;
        }
        System.out.printf("%s -> %d\n", varname, misses);
        cachemisses.put(varname, misses);
    }

    @Override
    public void exitArrayAccess(LoopNestParser.ArrayAccessContext ctx) {
        // System.out.println("arrayAccess");
        String varname = ctx.children.get(0).getText();
        HashMap<String, Integer> indices_to_dimindex = new HashMap<String, Integer>();
        for(int i = 2; i < ctx.getChildCount(); i += 3) {
            indices_to_dimindex.put(ctx.children.get(i).getText(), (i-2) / 3);
        }
        processArrayAccess(varname, indices_to_dimindex);
    }

    @Override
    public void exitArrayAccess_lfno_primary(LoopNestParser.ArrayAccess_lfno_primaryContext ctx) {
        // System.out.println("arrayAccess_lfno_primary");
        String varname = ctx.children.get(0).getText();
        HashMap<String, Integer> indices_to_dimindex = new HashMap<String, Integer>();
        for(int i = 2; i < ctx.getChildCount(); i += 3) {
            indices_to_dimindex.put(ctx.children.get(i).getText(), (i-2) / 3);
        }
        processArrayAccess(varname, indices_to_dimindex);
    }

}
