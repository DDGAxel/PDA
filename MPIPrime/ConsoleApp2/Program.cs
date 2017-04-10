using MPI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp2
{
    class Program
    {
        static void Main(string[] args)
        {
            string str = Console.ReadLine();
            int n = int.Parse(str);

            using (new MPI.Environment(ref args))
            {
                Intracommunicator comm = Communicator.world;
                comm.Barrier();
                for(int j = 1; j<= n; j++)
                {
                    if (Check_Prime(j))
                    {
                            Console.WriteLine(j + "is a prime number");
                    }
                }
            }
        }
        static bool Check_Prime(int number)
        {
            int i;
            for (i = 2; i <= number - 1; i++)
            {
                if (number % i == 0)
                {
                    return false;
                }
            }
            if (i == number)
            {
                return true;
            }
            return false;
        }
    }
}
