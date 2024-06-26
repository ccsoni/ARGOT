#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

#define MAXPATHLEN (1024)

void make_directory(char *directory_name)
{
  mode_t mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
  char *cwd;

  static char cwd_path[MAXPATHLEN];

  cwd = getcwd(cwd_path, sizeof(cwd_path));

  strcat(cwd_path,"/");
  strcat(cwd_path, directory_name);

  mkdir(cwd_path, mode);
}
