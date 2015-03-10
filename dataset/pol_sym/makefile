LIB = libsim.so

SRC =   sim.c

CC = gcc

OBJ = $(SRC:%.c=%.o)

all: $(LIB)

re: fclean all

$(LIB): $(OBJ)
	$(CC) -shared -fPIC $(OBJ) -o $(LIB)

clean:
	$(RM) $(OBJ)

fclean: clean
	$(RM) $(LIB)
